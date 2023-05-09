import torch
import torch.nn as nn
from module import BiLSTM
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = nn.Sequential(
            # input 1, 32, 160

            # in_channels, out_channels, kernel_size, stride=1, padding=0,
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x80

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x40

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),  # 256x8x40

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            # kernel_size stride padding
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x41

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 512x4x41

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x42

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # 512x1x25

        self.rnn = nn.Sequential(
            BiLSTM(512, nh, nh),
            BiLSTM(nh, nh, nh))

    def forward(self, x):
        conved = self.cnn(x)
        batch_size, channel, height, width = conved.size()
        assert height == 1, "the height of conv must be 1"
        conved = conved.squeeze(2)
        conved = conved.permute(2, 0, 1)  # [width, batch_size, channel]
        output = self.rnn(conved)
        return output
