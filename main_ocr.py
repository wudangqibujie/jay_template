from torch.utils.data import DataLoader
from data_loader.captcha_dataloader import Captcha2Data
from torchvision.transforms import Compose, ToTensor, Resize
from model.cnn_model.crnn import CRNN
from model.metric import captcha_acc
import torch


def target_transform(vec):
    rslt = []
    for ix, i in enumerate(vec):
        if i == 1:
            rslt.append(ix)
    return rslt

IMG_HEIGHT = 32
IMG_WIDTH = 160
NC = 1
NH = 256
train_folder = '../../dataset/train'
valid_folder = '../../dataset/test'
train_dataset = Captcha2Data(train_folder, transform=Compose([Resize([IMG_HEIGHT, IMG_WIDTH]), ToTensor()]), img_convert_mode='L', target_transform=target_transform)
valid_dataset = Captcha2Data(valid_folder, transform=Compose([Resize([IMG_HEIGHT, IMG_WIDTH]), ToTensor()]), img_convert_mode='L', target_transform=target_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)


encoder = CRNN(IMG_HEIGHT, NC, NH)

for X, y in train_dataloader:
    encoder_out = encoder(X)
    print(encoder_out.shape)
    break


