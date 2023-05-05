import numpy

from data_loader.captcha_dataloader import CaptchaDataset
from torch.utils.data import DataLoader
from model.captcha_model import CNN
from utils import prepare_device
import torch
from model.loss import multiclass_hinge_loss, multiclass_cross_entropy
from model.metric import multiclass_accuracy, multiclass_ele_accuracy
from torch.optim.lr_scheduler import StepLR
from trainer.jay_trainer import Trainer
from config import captcha_task_config


model_config = captcha_task_config.ModelConfig(60, 160, 1, len(captcha_task_config.ALL_CHAR_SET))
captcha_config = captcha_task_config.CaptchaConfig(1, captcha_task_config.ALL_CHAR_SET, len(captcha_task_config.ALL_CHAR_SET), 60, 160)

train_config = {
    'epochs': 100,
    'save_dir': "saveed/",
    'save_period': 5,
    'verbosity': 2,
    'monitor': "min val_loss",
    'early_stop': 10,
    'tensorboard': True,
    'log_step': 10,
}

config = {
    "save_dir": "model_log",
    "resume": None,
    "trainer": train_config,
    "captcha_config": captcha_config,
    "model_config": model_config,
}

model = CNN(model_config)
device, device_ids = prepare_device(2)
print(device, device_ids)
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
print(model)
criterion = multiclass_cross_entropy
metrics = [multiclass_accuracy, multiclass_ele_accuracy]
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.0005)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

train_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TRAIN_DATASET_PATH, captcha_config), batch_size=64,
                              shuffle=True)
valid_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TEST_DATASET_PATH, captcha_config), batch_size=64)

trainer = Trainer(model,
                  criterion,
                  metrics,
                  optimizer,
                  config=config,
                  device=device,
                  train_data_loader=train_dataloader,
                  valid_data_loader=valid_dataloader,
                  lr_scheduler=lr_scheduler,
                  len_epoch=None,
                  log_step=50)
trainer.train()
#
# def multiclass_accuracy(predict_labels, labels):
#     predict_labels = torch.split(predict_labels, 38, dim=1)
#     labels = torch.split(labels, 38, dim=1)
#     mk = []
#     for predict_label, label in zip(predict_labels, labels):
#         mk.append(torch.argmax(predict_label, dim=1) == torch.argmax(label, dim=1))
#     mk_con = torch.sum(torch.sum(torch.stack(mk, dim=1), dim=1) == 4).item()
#     return round(mk_con / labels[0].size(0), 4)
#
# for data, taregt in train_dataloader:
#     out = model(data)
#     acc = multiclass_accuracy(out, taregt)
#     print(acc)


