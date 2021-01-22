# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:30 下午
# @Author  : jeffery
# @FileName: project_utils.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:


import yaml
import json
from pathlib import Path
from collections import OrderedDict
import random
import os
import numpy as np
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('r',encoding='utf8') as handle:
        return yaml.load(handle,Loader=yaml.Loader)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_yaml(content,fname):
    fname = Path(fname)
    with fname.open('w',encoding='utf8') as handle:
        yaml.dump(content,handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)



def seed_everything(seed=1123):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True