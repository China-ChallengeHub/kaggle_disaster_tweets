# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 5:01 下午
# @Author  : jeffery
# @FileName: run_rnn.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:



import torch
import numpy as np
from model import makeModel, makeLoss, makeMetrics, makeOptimizer,makeLrSchedule
from data_process import makeDataLoader
from utils import ConfigParser,seed_everything
from trainer.trainer_softmax import Trainer
import yaml




def main(config):
    logger = config.get_logger('train')
    # fix random seeds for reproducibility
    seed_everything(seed=config.config['seed'])
    metric_bests = []
    # logger = config.get_logger('train')
    for i,train_dataloader, valid_dataloader, test_dataloader in makeDataLoader(config):

        model = makeModel(config)
        # logger.info(model)

        criterion = makeLoss(config)
        metrics = makeMetrics(config)

        optimizer = makeOptimizer(config, model)
        lr_scheduler = makeLrSchedule(config, optimizer, train_dataloader)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          i_fold=i,
                          data_loader=train_dataloader,
                          valid_data_loader=valid_dataloader,
                          test_data_loader=test_dataloader,
                          lr_scheduler=lr_scheduler)

        trainer.train()
        metric_bests.append(trainer.mnt_best)
    logger.info('metric scores:{}'.format(metric_bests))
    logger.info('metric mean score: {}'.format(sum(metric_bests) / float(len(metric_bests))))

def run(config_fname):
    with open(config_fname, 'r', encoding='utf8') as f:
        config_params = yaml.load(f, Loader=yaml.Loader)
        config_params['config_file_name'] = config_fname

    config = ConfigParser.from_args(config_params)
    main(config)

if __name__ == '__main__':

    # run('configs/transformer_pure_softmax.yml')
    # run('configs/transformer_cnn_softmax.yml')
    run('configs/transformer_rnn_softmax.yml')