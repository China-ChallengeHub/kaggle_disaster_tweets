# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:25 下午
# @Author  : jeffery
# @FileName: models.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from base import BaseModel
from base import BaseModel
from transformers import AlbertModel, AlbertConfig, AutoConfig, AutoModel
from torch import nn
from torch.nn import functional as F
import pickle
import torch
import numpy as np
import os
from pathlib import Path

class TransformersModel(BaseModel):

    def __init__(self, transformer_model, cache_dir, force_download, is_train,dropout, class_num):
        super(TransformersModel, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # 是否对transformers参数进行训练
        self.transformer_model.training = is_train
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.transformer_config.hidden_size, class_num)

    def forward(self, input_ids, attention_masks, text_lengths):
        transformer_output = self.transformer_model(input_ids, attention_mask=attention_masks)
        out = self.fc(self.dropout(torch.mean(transformer_output[0],dim=1)))
        return out.squeeze(), transformer_output[1]

class TransformersCNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, n_filters, filter_sizes, dropout,
                 is_train, class_num):
        super(TransformersCNN, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # 是否对transformers参数进行训练
        self.transformer_model.training = is_train
        # cnn
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (k, self.transformer_model.config.to_dict()['hidden_size'])) for k in
             filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(n_filters * len(filter_sizes), class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, int(x.size(2))).squeeze(2)
        return x

    def forward(self, input_ids, attention_masks, text_lengths):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.transformer_model(input_ids, attention_mask=attention_masks)
        encoder_out = self.dropout(encoder_out)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out_embedding = self.dropout(out)
        out = self.fc(out_embedding)
        return out, out_embedding



class TransformersRNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, rnn_type, hidden_dim, n_layers, bidirectional,
                 batch_first, dropout,is_train, class_num):
        super(TransformersRNN, self).__init__()

        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # 是否对transformers参数进行训练
        self.transformer_model.training = is_train

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.transformer_model.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 + self.transformer_config.hidden_size, class_num)

    def forward(self, input_ids, attention_masks, text_lengths):

        # text = [batch size,sent len]
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        sentence_out, cls = self.transformer_model(input_ids, attention_mask=attention_masks)
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(self.dropout(sentence_out))
        else:
            output, (hidden, cell) = self.rnn(self.dropout(sentence_out))

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = torch.cat([self.dropout(output + hidden),self.dropout(cls)],dim=1)
        out = self.fc(fc_input)

        return out,fc_input
