import torch
import torch.nn as nn
from module import BidirectionalLSTM, AttnDecoderRNN
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class CNN(nn.Module):
    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True))

    def forward(self, input):
        return self.cnn(input)


class CRNN(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(CRNN, self).__init__()
        self.cnn = CNN(channel_size)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result