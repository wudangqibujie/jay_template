from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from random import shuffle


class CaptchaDataset(Dataset):
    def __init__(self, folder, captcha_config):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        shuffle(self.train_image_file_paths)
        self.transform = transforms.Compose([
                        # transforms.ColorJitter(),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        self.label_encoder = self.encode(captcha_config)

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_encoder(image_name.split('_')[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        return image, label

    @staticmethod
    def encode(captcha_setting):
        def warapper(text):
            vector = np.zeros(captcha_setting.ALL_CHAR_SET_LEN * captcha_setting.MAX_CAPTCHA, dtype=float)
            for i, c in enumerate(text):
                idx = i * captcha_setting.ALL_CHAR_SET_LEN + CaptchaDataset.char2pos(c)
                vector[idx] = 1.0
            return vector
        return warapper

    @staticmethod
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

# train_dataloader = DataLoader(CaptchaDataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform),
#                               batch_size=64, shuffle=True)
# valid_dataloader = DataLoader(CaptchaDataset(captcha_setting.TEST_DATASET_PATH, transform=transform), shuffle=True)
# predict_dataloader = DataLoader(CaptchaDataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform), shuffle=True)
