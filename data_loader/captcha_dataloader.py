import torch
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
                        transforms.Grayscale(),
                        transforms.ToTensor(),
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


source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)


def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split('.')[0].split('_')[0]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    return samples

class Captcha2Data(Dataset):
    def __init__(self,
                 data_path,
                 num_class=36,
                 num_char=4,
                 transform=None,
                 target_transform=None,
                 alphabet=alphabet,
                 img_convert_mode='RGB'):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.img_convert_mode = img_convert_mode
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(data_path, alphabet, num_class, num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img_path, target = self.samples[item]
        img = Image.open(img_path).convert(self.img_convert_mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)