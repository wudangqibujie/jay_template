import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from features.tabular_features.base_feature import BaseFeatureEncoder
import json

folder = Path('../data/movielens/train')
csvs = [folder / c for c in os.listdir(folder)]
dff = pd.read_csv(csvs[0])


class VarLenFeaturesEncoder:
    def __init__(self, varlen_col):
        self.varlen_col = varlen_col
        self.max_length = None
        self.new_cols = None
        self.encoder = None
        self.vocab_size = None

    def fit(self, df):
        self.max_length = int(df[self.varlen_col].apply(lambda x: len(x.split("|"))).max())
        self.new_cols = [f"varlen_{self.varlen_col}_{i}" for i in range(self.max_length)]
        df[f"varlen_{self.varlen_col}_len"] = df[self.varlen_col].apply(lambda x: len(x.split("|")))
        df[self.new_cols] = df[self.varlen_col].apply(lambda x: pd.Series(x.split("|")))
        df[self.new_cols] = df[self.new_cols].fillna("<UNK>")
        total_series = pd.concat([df[col] for col in self.new_cols])
        self.encoder = LabelEncoder()
        self.encoder.fit(total_series)
        self.vocab_size = len(self.encoder.classes_)

    def transform(self, df):
        df = pd.DataFrame(df)
        df[f"varlen_{self.varlen_col}_len"] = df[self.varlen_col].apply(lambda x: len(x.split("|")))
        df[self.new_cols] = df[self.varlen_col].apply(lambda x: pd.Series(x.split("|")))
        df[self.new_cols] = df[self.new_cols].fillna("<UNK>")
        for col in self.new_cols:
            df[col] = self.encoder.transform(df[col])
        return df[self.new_cols]


class FeatureEncoder(BaseFeatureEncoder):
    def __init__(self, sparse_cols, dense_cols, varlen_cols, target_cols):
        super().__init__(sparse_cols, dense_cols, varlen_cols, target_cols)
        self.sparse_dataset_config = []
        self.dense_dataset_config = []
        self.varlen_dataset_config = []
        self.target_dataset_config = []

    def fit(self, df):
        self._sparse_vocab = {}
        for col in self.sparse_cols:
            lbe = LabelEncoder()
            lbe.fit_transform(df[col])
            self._sparse_vocab[col] = len(lbe.classes_)
            self._sparse_transformers[col] = lbe
        for col in self.dense_cols:
            mms = MinMaxScaler()
            mms.fit_transform(df[col].values.reshape(-1, 1))
            self._dense_transformers[col] = mms
        var_encoder = VarLenFeaturesEncoder(self.varlen_cols[0])
        var_encoder.fit(df)
        self._varlen_transformers[self.varlen_cols[0]] = var_encoder

    def encode(self, df):
        for col, transform in self._sparse_transformers.items():
            df[f"lbe_{col}"] = transform.transform(df[col])
            self.sparse_dataset_config.append({"name": f"lbe_{col}",
                                               "vocab_size": self._sparse_vocab[col],
                                               "embed_dim": 8})
        for col, transform in self._dense_transformers.items():
            df[f"scale_{col}"] = transform.transform(df[col].values.reshape(-1, 1))
            self.dense_dataset_config.append({"name": f"scale_{col}"})
        for col, transform in self._varlen_transformers.items():
            df[transform.new_cols] = transform.transform(df[col])
            self.varlen_dataset_config.append({"name": f"varlen_{col}",
                                               "cols": transform.new_cols,
                                               "type": "sparse",
                                               "max_length": transform.max_length,
                                               "vocab_size": transform.vocab_size,
                                               "embed_dim": 8})
        df = self.preprocess(df)
        self.target_dataset_config.append({"name": "if_high_rated",
                                           "task_type": "binary_classification"})
        return df

    def preprocess(self, df):
        df[f'if_high_rated'] = (df[self.target_cols[0]] > 3).astype(int)
        return df

    def create_dataset_config(self, config_path):
        config = dict()
        config['dense'] = self.dense_dataset_config
        config['sparse'] = self.sparse_dataset_config
        config['varlen'] = self.varlen_dataset_config
        config['target'] = self.target_dataset_config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


# mvlen_sparse_cols = ['user_id', 'movie_id', 'gender', 'occupation', 'zip']
# mvlen_dense_cols = ['age']
# mvlen_var_cols = ['genres']
# mvlen_target_cols = ['rating']
# featureEncoder = FeatureEncoder(mvlen_sparse_cols, mvlen_dense_cols, mvlen_var_cols, mvlen_target_cols)
# featureEncoder.fit(dff)
# featureEncoder.preprocess(featureEncoder.encode(dff))
# featureEncoder.create_dataset_config("config/movielens_dataset.json")
# featureEncoder.save('../data/movielens/feature_encoder.pkl')
#
#
# featureEncoder = FeatureEncoder.load('../data/movielens/feature_encoder.pkl')
# df1 = pd.read_csv(csvs[1])
# df1 = featureEncoder.encode(df1)
# df1 = featureEncoder.preprocess(df1)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


data = pd.read_csv("mini_data/tabular_csv/movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", ]
target = ['rating']
# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# preprocess the sequence feature
key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
# 2.count #unique features for each sparse field and generate feature config for sequence feature
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                          for feat in sparse_features]
varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
    key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# 3.generate input data for model
model_input = {name: data[name] for name in sparse_features}
model_input["genres"] = genres_list
print(model_input)