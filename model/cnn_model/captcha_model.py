import torch
import torch.nn as nn
from base import BaseModel


class CNN(BaseModel):
    def __init__(self, model_config):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((model_config.image_width//8) * (model_config.image_height//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, model_config.max_captcha * model_config.all_char_set_len),
        )
        # self.rfc = [nn.Linear(1024, model_config.all_char_set_len) for _ in range(model_config.max_captcha)]

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = [rfc(out) for rfc in self.rfc]
        # out = torch.concat(out, dim=1)
        out = self.rfc(out)
        return out


class CaptchaModel(BaseModel):
    def __init__(self, num_class=36, num_char=4):
        super(CaptchaModel, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # batch*3*180*100
            nn.Conv2d(3, 16, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # batch*16*90*50
            nn.Conv2d(16, 64, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # batch*64*45*25
            nn.Conv2d(64, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*22*12
            nn.Conv2d(512, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*11*6
        )
        self.fc = nn.Linear(512 * 10 * 3, self.num_class * self.num_char)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 10 * 3)
        x = self.fc(x)
        return x

