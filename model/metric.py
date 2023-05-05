import torch
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


class MetricCollect:
    def __init__(self, metric_funcs, max_cum_num=None):
        self.metric_funcs = metric_funcs
        self.max_cum_num = max_cum_num
        self._labels = []
        self._predicts = []
        self.reset()
        self._cum_num = 0

    def reset(self):
        self._labels = []
        self._predicts = []
        self._cum_num = 0

    def update(self, batch_labels, batch_predicts):
        if self.max_cum_num is not None and self._cum_num >= self.max_cum_num:
            return
        self._labels.append(batch_labels)
        self._predicts.append(batch_predicts)
        self._cum_num += batch_labels.shape[0]

    def cal_metrics(self):
        assert self._cum_num != 0
        log = {"cum_num": self._cum_num}
        labels = torch.concat(self._labels, dim=0)
        predicts = torch.concat(self._predicts, dim=0)
        for metric_func in self.metric_funcs:
            log[metric_func.__name__] = metric_func(labels, predicts)
        return log

    @property
    def rslt(self):
        return self.cal_metrics()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def auc(output, target):
    return roc_auc_score(target, output.detach().numpy())


def ks(output, target):
    FPR, TPR, _ = roc_curve(target, output.detach().numpy())
    return abs(FPR - TPR).max()


# def multiclass_accuracy(predict_labels, labels):
#     predict_labels = torch.split(predict_labels, 38, dim=1)
#     labels = torch.split(labels, 38, dim=1)
#     mk = []
#     for predict_label, label in zip(predict_labels, labels):
#         mk.append(torch.argmax(predict_label, dim=1) == torch.argmax(label, dim=1))
#     mk_con = torch.sum(torch.sum(torch.stack(mk, dim=1), dim=1) == 4).item()
#     return round(mk_con / labels[0].size(0), 4)
#
#
# def multiclass_ele_accuracy(predict_labels, labels):
#     predict_labels = torch.split(predict_labels, 38, dim=1)
#     labels = torch.split(labels, 38, dim=1)
#     mk = []
#     for predict_label, label in zip(predict_labels, labels):
#         mk.append(torch.argmax(predict_label, dim=1) == torch.argmax(label, dim=1))
#     mk_con = torch.sum(torch.stack(mk, dim=1), dim=1).item()
#     return round(mk_con / (labels[0].size(0) * 4), 4)

def multiclass_accuracy(predict_labels, labels):
    ll_s = torch.sort(torch.topk(predict_labels, 1)[1], dim=1)[0]
    kk_s = torch.sort(torch.topk(labels, 1)[1], dim=1)[0]
    correct_num = torch.sum(torch.sum(torch.eq(ll_s, kk_s), dim=1) == 1).item()
    total_num = labels.size(0)
    return correct_num / total_num


def multiclass_ele_accuracy(predict_labels, labels):
    ll_s = torch.sort(torch.topk(predict_labels, 1)[1], dim=1)[0]
    kk_s = torch.sort(torch.topk(labels, 1)[1], dim=1)[0]
    total_ele_num = labels.size(0) * 1
    correct_ele_num = torch.sum(torch.eq(ll_s, kk_s)).item()
    return round(correct_ele_num / total_ele_num, 4)
