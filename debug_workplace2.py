from torch.utils.data import DataLoader
from data_loader.captcha_dataloader import Captcha2Data
from torchvision.transforms import Compose, ToTensor
from model.cnn_model.captcha_model import CaptchaModel
from model.metric import captcha_acc
import torch


train_folder = '../../dataset/train'
valid_folder = '../../dataset/test'
train_dataset = Captcha2Data(train_folder, transform=Compose([ToTensor()]))
valid_dataset = Captcha2Data(valid_folder, transform=Compose([ToTensor()]))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)

model = CaptchaModel()
optimzer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MultiLabelSoftMarginLoss()
for i, (image, label) in enumerate(train_dataloader):
    image = torch.autograd.Variable(image)
    label = torch.autograd.Variable(label)
    label = label.reshape(64, -1)
    output = model(image)
    print(output.shape, label.shape)
    loss = criterion(output, label)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    print(output.shape)
    acc = captcha_acc(output, label)
    print(acc)
    break


