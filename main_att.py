from config.captcha_task_config import CaptchaConfig
from data_loader.captcha_dataloader import CaptchaDataset
from torch.utils.data import DataLoader
from model.captcha_model import CNN
from utils import prepare_device
import torch
import os
from utils.util import read_json
from model.loss import multiclass_hinge_loss, multiclass_cross_entropy
from model.metric import multiclass_accuracy, multiclass_ele_accuracy
from torch.optim.lr_scheduler import StepLR
from trainer.jay_trainer import Trainer
from config import captcha_task_config

NUMBER = [str(i) for i in range(10)]
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBER + ALPHABET
TRAIN_DATASET_PATH = '../../dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = '../../dataset' + os.path.sep + 'test'

captchaConfig = CaptchaConfig(max_captcha=4,
                              number_char=NUMBER,
                              alphabet_char=ALPHABET,
                              image_height=60,
                              image_width=160,
                              base_config=read_json('config/captcha_config.json'),
                              train_dataset_path=TRAIN_DATASET_PATH,
                              test_dataset_path=TEST_DATASET_PATH)
train_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TRAIN_DATASET_PATH,
                                             captchaConfig.captcha_info),
                              batch_size=64,
                              shuffle=True)
valid_dataloader = DataLoader(CaptchaDataset(captcha_task_config.TEST_DATASET_PATH,
                                             captchaConfig.captcha_info),
                              batch_size=64)

for X, y in train_dataloader:
    print(X.shape, y.shape)
    break