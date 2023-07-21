import pickle
from abc import ABC, ABCMeta, abstractmethod


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

    def preprocess(self, df):
        pass

    def encode(self, df):
        for col, transform in self._sparse_transformers.items():
            df[f"lbe_{col}"] = transform.transform(df[col])
        for col, transform in self._dense_transformers.items():
            df[f"scale_{col}"] = transform.transform(df[col].values.reshape(-1, 1))
        print(df.columns)
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

    def create_dataset_config(self, config_path):
        pass


class BaseFeatureLabelInfo(metaclass=ABCMeta):
    def __init__(self):
        pass

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
    def from_json(cls):
        pass

    @classmethod
    def from_yaml(cls):
        pass




