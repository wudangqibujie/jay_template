import torch
import torch.nn as nn
from base import BaseModel



class TabBaseModel(BaseModel):
    def __init__(self):
        super(TabBaseModel, self).__init__()


class DeepFM(BaseModel):
    def __init__(self):
        super(DeepFM, self).__init__()
        self.embedding_layer = []
        self.fm_layer = []
        self.deep_layer = []

    def forward(self, x):
        pass
