from config.captcha_task_config import CaptchaConfig
from dataset.captcha_dataloader import CaptchaDataset
from torch.utils.data import DataLoader
from model.captcha_model import CNN
from utils import prepare_device
import torch
import os
from utils.util import read_json
from model.loss import multiclass_cross_entropy
from model.metric import multiclass_accuracy, multiclass_ele_accuracy
from torch.optim.lr_scheduler import StepLR
from trainer.jay_trainer import Trainer
from config import captcha_task_config


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBER + ALPHABET
TRAIN_DATASET_PATH = '../../dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = '../../dataset' + os.path.sep + 'test'
# model_config = captcha_task_config.ModelConfig(60, 160, 1, len(captcha_task_config.ALL_CHAR_SET))
# captcha_config = captcha_task_config.CaptchaConfig(1, captcha_task_config.ALL_CHAR_SET, len(captcha_task_config.ALL_CHAR_SET), 60, 160)
# train_config = {
#     'epochs': 100,
#     'save_dir': "saved/",
#     'save_period': 5,
#     'verbosity': 2,
#     'monitor': "min val_loss",
#     'early_stop': 10,
#     'tensorboard': True,
#     'log_step': 10,
# }
#
# config = {
#     "save_dir": "../model_log",
#     "resume": None,
#     "trainer": train_config,
#     "captcha_config": captcha_config,
#     "model_config": model_config,
# }

captchaConfig = CaptchaConfig(max_captcha=4,
                              number_char=NUMBER,
                              alphabet_char=ALPHABET,
                              image_height=60,
                              image_width=160,
                              base_config=read_json('config/captcha_config.json'),
                              train_dataset_path=TRAIN_DATASET_PATH,
                              test_dataset_path=TEST_DATASET_PATH)

model = CNN(
    captchaConfig.image_height,
    captchaConfig.image_width,
    captchaConfig.max_captcha,
    captchaConfig.all_char_set_len
)
logger = captchaConfig.get_logger("main")
device, device_ids = prepare_device(2)
logger.info(f"device: {device}, device_ids: {device_ids}")
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
logger.info(model)
criterion = multiclass_cross_entropy
metrics = [multiclass_accuracy, multiclass_ele_accuracy]
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.0005)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

train_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TRAIN_DATASET_PATH,
                                             captchaConfig.captcha_info),
                              batch_size=64,
                              shuffle=True)
valid_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TEST_DATASET_PATH,
                                             captchaConfig.captcha_info),
                              batch_size=64)

logger.info(str(captchaConfig.get_config))

trainer = Trainer(model,
                  criterion,
                  metrics,
                  optimizer,
                  config=captchaConfig,
                  device=device,
                  train_data_loader=train_dataloader,
                  valid_data_loader=valid_dataloader,
                  lr_scheduler=lr_scheduler,
                  len_epoch=None,
                  log_step=50)
trainer.train()
