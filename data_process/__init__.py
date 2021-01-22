# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:22 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import data_process.data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
import torch

def makeDataLoader(config):
    # setup data_set, data_process instances
    test_set = config.init_obj('test_set', module_data_process)
    test_dataloader = module_dataloader.DataLoader(test_set, batch_size=test_set.batch_size,
                                                   num_workers=test_set.num_workers, collate_fn=test_set.collate_fn)

    if not config['k_fold'] > 0:
        train_set = config.init_obj('train_set', module_data_process)
        valid_set = config.init_obj('valid_set', module_data_process)

        train_dataloader = module_dataloader.DataLoader(train_set, batch_size=train_set.batch_size,
                                                        num_workers=train_set.num_workers, collate_fn=train_set.collate_fn)
        valid_dataloader = module_dataloader.DataLoader(valid_set, batch_size=valid_set.batch_size,
                                                        num_workers=valid_set.num_workers, collate_fn=valid_set.collate_fn)

        yield 0,train_dataloader, valid_dataloader, test_dataloader
    else:
        logger = config.get_logger('train')
        logger.info('making {} fold data'.format(config['k_fold']))
        all_set = config.init_obj('all_set', module_data_process)
        for i,train_index,valid_index in all_set.make_k_fold_data(config['k_fold']):
            train_set = torch.utils.data.dataset.Subset(all_set, train_index)
            valid_set = torch.utils.data.dataset.Subset(all_set,valid_index)
            train_dataloader = module_dataloader.DataLoader(train_set, batch_size=all_set.batch_size,
                                                            num_workers=all_set.num_workers,drop_last=True,
                                                            collate_fn=all_set.collate_fn)
            valid_dataloader = module_dataloader.DataLoader(valid_set, batch_size=all_set.batch_size,
                                                            num_workers=all_set.num_workers,
                                                            collate_fn=all_set.collate_fn)

            yield i,train_dataloader, valid_dataloader, test_dataloader