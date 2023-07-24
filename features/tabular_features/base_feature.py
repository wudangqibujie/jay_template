import pickle
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
import json
import yaml


@dataclass
class SparseFeature:
    name: str
    vocab_size: int
    embed_dim: int


@dataclass
class DenseFeature:
    name: str


@dataclass
class VarLenFeature:
    name: str
    cols: list
    type: str
    max_length: int
    vocab_size: int
    embed_dim: int


class BaseFeatureEncoder(metaclass=ABCMeta):
    def __init__(self, sparse_cols, dense_cols, varlen_cols, target_cols):
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        self.varlen_cols = varlen_cols
        self.target_cols = target_cols
        self._sparse_transformers = {}
        self._dense_transformers = {}
        self._varlen_transformers = {}
        self._target_transformers = {}

    @abstractmethod
    def fit(self, df):
        raise NotImplementedError

    def encode(self, df):
        for col, transform in self._sparse_transformers.items():
            df[f"lbe_{col}"] = transform.transform(df[col])
        for col, transform in self._dense_transformers.items():
            df[f"scale_{col}"] = transform.transform(df[col].values.reshape(-1, 1))
        for col, transform in self._varlen_transformers.items():
            df[transform.new_cols] = transform.transform(df[col])
        for col, transform in self._target_transformers.items():
            df[f"target_{col}"] = transform.transform(df[col])
        return df

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class BaseFeatureLabelInfo(metaclass=ABCMeta):
    def __init__(self, sparse_features, dense_features, varlen_features, target_features):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.varlen_features = varlen_features
        self.target_features = target_features

    def step_filter(self):
        pass

    def step_transform(self):
        pass

    def __str__(self):
        pass

    @abstractmethod
    def create_batch(self):
        raise NotImplementedError

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        dense_features = [DenseFeature(**feature) for feature in config['dense']]
        sparse_features = [SparseFeature(**feature) for feature in config['sparse']]
        varlen_features = [VarLenFeature(**feature) for feature in config['varlen']]
        target_features = [DenseFeature(**feature) for feature in config['target']]
        return cls(sparse_features, dense_features, varlen_features, target_features)

    @classmethod
    def from_yaml(cls, path):
        pass



