from base.base_config import BaseConfig
from config.captcha_task_config import CaptchaConfig
from utils.util import read_json
import os

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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

total_config = captchaConfig.get_config
print(total_config)