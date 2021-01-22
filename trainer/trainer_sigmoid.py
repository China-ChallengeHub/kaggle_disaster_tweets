# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:29 下午
# @Author  : jeffery
# @FileName: trainer_sigmoid.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:


from utils import inf_loop, MetricTracker
from base import BaseTrainer
import torch
import numpy as np
import model.adversarial as module_adversarial
import model.metric as module_mertric
import json

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config,i_fold, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.i_fold = i_fold
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.zero_grad()
        self.train_metrics.reset()
        adv_train = self.config.init_obj('adversarial_training',module_adversarial,model=self.model)
        K = 3
        for batch_idx, data in enumerate(self.data_loader):
            self.model.train()
            ids,texts, input_ids, attention_masks, text_lengths, labels = data

            if 'cuda' == self.device.type:
                input_ids = input_ids.cuda(self.device)
                attention_masks = attention_masks.cuda(self.device)
                labels = labels.cuda(self.device)

            preds, cls_embedding = self.model(input_ids, attention_masks,text_lengths)
            loss = self.criterion[0](preds, labels)
            # 损失截断
            loss_zeros = torch.zeros_like(loss)
            loss = torch.where(loss > float(self.config.config['loss']['loss_cut']), loss, loss_zeros)
            loss.backward()
            if self.config.config['trainer']['is_adversarial_training'] and self.config.config['adversarial_training']['type']=='FGM': # 对抗训练
                adv_train.attack()
                adv_preds,adv_cls_embedding = self.model(input_ids,attention_masks,text_lengths)
                adv_loss = self.criterion[0](adv_preds, labels)
                adv_loss.backward()
                adv_train.restore()
            elif self.config.config['trainer']['is_adversarial_training'] and self.config.config['adversarial_training']['type']=='PGD':
                adv_train.backup_grad()
                # 对抗训练
                for t in range(K):
                    adv_train.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        self.model.zero_grad()
                    else:
                        adv_train.restore_grad()
                    adv_preds, adv_cls_embedding= self.model(input_ids,attention_masks,text_lengths)
                    adv_loss = self.criterion[0](adv_preds, labels)
                    adv_loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                adv_train.restore()  # 恢复embedding参数

            if self.config.config['trainer']['clip_grad']: # 梯度截断
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.config['trainer']['max_grad_norm'])
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.model.zero_grad()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(preds, labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.3f} lr: {}'.format(epoch, self._progress(batch_idx),
                                                                           loss.item(),self.optimizer.param_groups[0]['lr']))
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.valid_data_loader:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                ids,texts, input_ids, attention_masks, text_lengths, labels = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda(self.device)
                    attention_masks = attention_masks.cuda(self.device)
                    labels = labels.cuda(self.device)
                preds,cls_embedding = self.model(input_ids, attention_masks,text_lengths)

                if self.add_graph:
                    input_model = self.model.module if (len(self.config.config['device_id']) > 1) else self.model
                    self.writer.writer.add_graph(input_model,
                                                 [input_ids, attention_masks, text_lengths])
                    self.add_graph = False
                loss = self.criterion[0](preds, labels)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(preds, labels))

        log = self.valid_metrics.result()
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return log

    def _inference(self):
        """
        Inference after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        checkpoint = torch.load(self.best_path)
        self.logger.info("load best mode {} ...".format(self.best_path))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        ps = []
        ls = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                ids,texts, input_ids, attention_masks, text_lengths, labels = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda(self.device)
                    attention_masks = attention_masks.cuda(self.device)
                    labels = labels.cuda(self.device)
                preds,cls_embedding = self.model(input_ids, attention_masks,text_lengths)
                ps.append(preds)
                ls.append(labels)

        ps = torch.cat(ps,dim=0)
        ls = torch.cat(ls,dim=0)
        acc = module_mertric.binary_accuracy(ps,ls)
        self.logger.info('\toverall   acc :{}'.format(acc))

        result_file = self.test_data_loader.dataset.data_dir.parent / 'result' /'{}-{}-{}-{}-{}.jsonl'.format(
            self.config.config['experiment_name'],
             self.test_data_loader.dataset.transformer_model,
            self.config.config['k_fold'],self.i_fold,acc)

        if not result_file.parent.exists():
            result_file.parent.mkdir()

        result_writer = result_file.open('w')

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                ids,texts, input_ids, attention_masks, text_lengths, labels = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda(self.device)
                    attention_masks = attention_masks.cuda(self.device)
                preds,cls_embedding = self.model(input_ids, attention_masks,text_lengths)
                preds =torch.round(torch.sigmoid(preds)).cpu().detach().numpy()
                for pred, item_id, text in zip(preds, ids, texts):
                    result_writer.write(json.dumps({
                        "id": item_id,
                        "text": text,
                        "labels": int(pred)
                    }, ensure_ascii=False) + '\n')

            result_writer.close()
            self.logger.info('result saving to {}'.format(result_file))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
