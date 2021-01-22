# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 8:25 下午
# @Author  : jeffery
# @FileName: loss.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch.nn.functional as F

def ce_loss(output, target):

    return F.cross_entropy(output, target)


def binary_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target.float())

def label_smoothing_ce_loss(output, target, epsilon=0.1, reduction='mean', ignore_index=-100):
    n_classes = output.size()[-1]
    log_preds = F.log_softmax(output, dim=1)
    if reduction == 'sum':
        loss = -log_preds.sum()
    else:
        loss = -log_preds.sum(dim=1)
        if reduction == 'mean':
            loss = loss.mean()
    return loss * epsilon / n_classes + (1 - epsilon) * F.nll_loss(log_preds, target, reduction=reduction, ignore_index=ignore_index)

