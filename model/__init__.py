# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:24 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:


import torch
import transformers
import model.models as module_models
import model.loss as module_loss
import model.metric as module_metric

__all__ = ["makeModel", "makeLoss", "makeMetrics", "makeOptimizer","makeLrSchedule"]


def makeModel(config):
    return config.init_obj('model_arch', module_models,)


def makeLoss(config):
    return [getattr(module_loss, crit) for crit in config['loss']['losses']]


def makeMetrics(config):
    return [getattr(module_metric, met) for met in config['metrics']]


def makeOptimizer(config, model):
    no_decay = ["bias", "LayerNorm.weight"]

    transformers_parameters_with_decay = [p for n, p in model.transformer_model.named_parameters() if
                                          not any(nd in n for nd in no_decay)]
    transformers_parameters = [p for n, p in model.transformer_model.named_parameters() if
                               any(nd in n for nd in no_decay)]
    fc_parameters_with_decay = [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)]
    fc_parameters = [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)]

    if 'cnn' in config.config['model_arch']['type'].lower():
        convs_parameters_with_decay = [p for n, p in model.convs.named_parameters() if not any(nd in n for nd in no_decay)]
        convs_parameters = [p for n, p in model.convs.named_parameters() if any(nd in n for nd in no_decay)]
        fc_parameters_with_decay = fc_parameters_with_decay + convs_parameters_with_decay
        fc_parameters = fc_parameters + convs_parameters

    if 'rnn_type' in config.config['model_arch']['args']:
        rnn_parameters_with_decay = [p for n, p in model.rnn.named_parameters() if not any(nd in n for nd in no_decay)]
        rnn_parameters = [p for n, p in model.rnn.named_parameters() if any(nd in n for nd in no_decay)]
        fc_parameters_with_decay = fc_parameters_with_decay + rnn_parameters_with_decay
        fc_parameters = fc_parameters + rnn_parameters


    weight_decay = float(config.config['optimizer']['weight_decay'])
    transformer_lr = float(config.config['optimizer']['transformers_lr'])
    crf_lr = float(config.config['optimizer']['crf_lr'])
    fc_lr = float(config.config['optimizer']['fc_lr'])

    if 'crf' in config.config['model_arch']['type'].lower():
        crf_parameters_with_decay = [p for n, p in model.crf.named_parameters() if not any(nd in n for nd in no_decay)]
        crf_parameters = [p for n, p in model.crf.named_parameters() if any(nd in n for nd in no_decay)]

        optimizer = config.init_obj('optimizer', torch.optim, [
            {'params': transformers_parameters_with_decay, 'weight_decay': weight_decay, 'lr': transformer_lr},
            {'params': transformers_parameters, 'weight_decay': 0.0, 'lr': transformer_lr},
            {'params': fc_parameters_with_decay, 'weight_decay': weight_decay, 'lr': fc_lr},
            {'params': fc_parameters, 'weight_decay': 0.0, 'lr': fc_lr},
            {'params': crf_parameters_with_decay, 'weight_decay': weight_decay, 'lr': crf_lr},
            {'params': crf_parameters, 'weight_decay': 0.0, 'lr': crf_lr}
        ])
    else:
        optimizer = config.init_obj('optimizer', torch.optim, [
            {'params': transformers_parameters_with_decay, 'weight_decay': weight_decay, 'lr': transformer_lr},
            {'params': transformers_parameters, 'weight_decay': 0.0, 'lr': transformer_lr},
            {'params': fc_parameters_with_decay, 'weight_decay': weight_decay, 'lr': fc_lr},
            {'params': fc_parameters, 'weight_decay': 0.0, 'lr': fc_lr},
        ])

    return optimizer


def makeLrSchedule(config, optimizer, train_dataloader):
    t_total = len(train_dataloader) * config.config['trainer']['epochs']
    num_warmup_steps = int(config.config['lr_scheduler']['warmup_proportion'] * t_total)
    # lr_scheduler = config.init_obj('lr_scheduler', optimization.lr_scheduler, optimizer)
    lr_scheduler = config.init_obj('lr_scheduler', transformers.optimization, optimizer,num_warmup_steps=num_warmup_steps,
                                   num_training_steps=t_total)

    return lr_scheduler
