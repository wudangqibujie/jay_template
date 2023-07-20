from PIL import Image
from captcha.image import ImageCaptcha
from tensorflow.python.keras.callbacks import Callback
import random
from xgboost import XGBClassifier
import time
from multiprocessing import Manager
Manager().Queue()
# import captcha_setting
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
from deepctr_torch.models import MMOE
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_DATASET_PATH = '../../dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = '../../dataset' + os.path.sep + 'test'
PREDICT_DATASET_PATH = '../../dataset' + os.path.sep + 'predict'

print(os.getcwd())
print(ImageCaptcha)


def random_captcha():
    captcha_text = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 1000
    path = TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text+'_'+now+'.png'
        image.save(path  + os.path.sep +  filename)
        print('saved %d : %s' % (i+1,filename))