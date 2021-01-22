# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:22 下午
# @Author  : jeffery
# @FileName: data_process.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:



from base import BaseDataSet
from tqdm import tqdm
import torch
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InputExample:
    guid: Optional[str]
    text: str
    keyword:str
    location:str
    label: float


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    guid: Optional[str]
    input_ids: List[int]
    attention_mask: Optional[List[int]]
    label: float
    text: str
    sent_len: int


class DisasterDataset(BaseDataSet):

    def __init__(self, data_dir, file_name, cache_dir, shuffle, transformer_model, overwrite_cache, batch_size,
                 force_download, num_workers):
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformer_model = transformer_model
        self.feature_cache_file = Path(cache_dir) / self.data_dir.parent.name / (file_name.split('.')[0] + '.cache')
        super(DisasterDataset, self).__init__(transformer_model=transformer_model,
                                             overwrite_cache=overwrite_cache,
                                             force_download=force_download, cache_dir=cache_dir)

    def read_examples_from_file(self):
        input_file = self.data_dir / self.file_name

        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                yield InputExample(guid=json_line['id'], text=json_line['text'],keyword=json_line['keyword'],
                                   location=json_line['location'],label=float(json_line['label']))

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            text = example.text + example.keyword + example.location
            inputs = self.tokenizer.encode_plus(text, return_token_type_ids=False,
                                                return_attention_mask=True, return_length=True)

            features.append(InputFeatures(guid=example.guid, input_ids=inputs.data['input_ids'],
                                          attention_mask=inputs.data['attention_mask'],
                                          sent_len=inputs.data['length'], text=example.text,
                                          label=example.label))
        return features

    def collate_fn(self, datas: List[InputFeatures]):
        max_len = max([data.sent_len for data in datas])
        ids = []
        input_ids = []
        attention_masks = []
        text_lengths = []
        labels = []
        texts = []

        for data in datas:
            ids.append(data.guid)
            texts.append(data.text)
            input_ids.append(data.input_ids + [self.tokenizer.pad_token_id] * (max_len - data.sent_len))
            attention_masks.append(data.attention_mask + [0] * (max_len - data.sent_len))
            text_lengths.append(data.sent_len)
            labels.append(data.label)

        input_ids = torch.LongTensor(input_ids)
        attention_masks = torch.ByteTensor(attention_masks)
        text_lengths = torch.LongTensor(text_lengths)
        labels = torch.LongTensor(labels)
        return ids, texts, input_ids, attention_masks, text_lengths, labels

