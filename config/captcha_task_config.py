from dataclasses import dataclass
from base.base_config import BaseConfig
import os


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBER + ALPHABET
TRAIN_DATASET_PATH = '../../dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = '../../dataset' + os.path.sep + 'test'


@dataclass
class CaptchaInfo:
    MAX_CAPTCHA: int
    ALL_CHAR_SET: list
    ALL_CHAR_SET_LEN: int
    IMAGE_HEIGHT: int
    IMAGE_WIDTH: int

# @dataclass
# class ModelConfig(BaseConfig):
#     image_height: int
#     image_width: int
#     max_captcha: int
#     all_char_set_len: int

@dataclass
class CaptchaConfig(BaseConfig):
    max_captcha: int
    number_char: list
    alphabet_char: list
    image_height: int
    image_width: int
    base_config: dict
    train_dataset_path: str
    test_dataset_path: str
    task_id: str = None
    captcha_info: CaptchaInfo = None

    def __post_init__(self):
        self.all_char_set = self.number_char + self.alphabet_char
        self.all_char_set_len = len(self.all_char_set)
        # custom_config = self.__dict__.copy()
        self.captcha_info = CaptchaInfo(self.max_captcha, self.all_char_set, self.all_char_set_len, self.image_height, self.image_width)
        super().__init__(self.base_config, task_id=self.task_id)
        # self.update_config(custom_config)

