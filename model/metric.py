# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:25 下午
# @Author  : jeffery
# @FileName: metric.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    rounded_preds = torch.round(torch.sigmoid(preds)).squeeze()
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=-1)  # get the index of the max probability
    correct = max_preds.eq(y)
    return (correct.sum().cpu() / torch.FloatTensor([y.shape[0]])).item()
