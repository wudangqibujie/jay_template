from data_loader.ocr_dataloader import TextLineDataset
from data_loader.captcha_dataloader import Captcha2Data
from torchvision.transforms import Compose, ToTensor, Resize
from model.cnn_model.crnn import CRNN
from model.metric import captcha_acc
import torch


train_dataset = TextLineDataset('mini_data/ocr_data')