import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def multiclass_hinge_loss(output, target):
    cri = nn.MultiLabelSoftMarginLoss()
    return cri(output, target)


def multiclass_cross_entropy(output, target):
    return F.cross_entropy(output, target)