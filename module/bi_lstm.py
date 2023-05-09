import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiDeirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        # input, 为 length * batch * dim
        super(BiDeirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

