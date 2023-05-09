from config.captcha_task_config import CaptchaConfig
from config.captcha_task_config2 import BaseConfig
from data_loader.captcha_dataloader import CaptchaDataset
from torch.utils.data import DataLoader
from data_loader.captcha_dataloader import Captcha2Data
from utils import prepare_device
from torchvision.transforms import Compose, ToTensor
import torch
from model.cnn_model.captcha_model import CaptchaModel
import os
from utils.util import read_json
from model.metric import captcha_acc
from torch.optim.lr_scheduler import StepLR
from trainer.jay_trainer import Trainer
from config import captcha_task_config


train_folder = '../../dataset/train'
valid_folder = '../../dataset/test'
train_dataset = Captcha2Data(train_folder, transform=Compose([ToTensor()]))
valid_dataset = Captcha2Data(valid_folder, transform=Compose([ToTensor()]))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)


config = BaseConfig.from_json('config/captcha_config.json')
model = CaptchaModel()
logger = config.get_logger("main")
device, device_ids = prepare_device(2)
logger.info(f"device: {device}, device_ids: {device_ids}")
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
logger.info(model)

criterion = torch.nn.MultiLabelSoftMarginLoss()
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimzer = torch.optim.Adam(trainable_params, lr=0.001)
metrics = [captcha_acc]
lr_scheduler = StepLR(optimzer, step_size=10, gamma=0.1)
logger.info(str(config.get_config))
trainer = Trainer(model,
                  criterion,
                  metrics,
                  optimzer,
                  config=config,
                  device=device,
                  train_data_loader=train_dataloader,
                  valid_data_loader=valid_dataloader,
                  lr_scheduler=lr_scheduler,
                  len_epoch=None,
                  log_step=50)
trainer.train()
