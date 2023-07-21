import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
from features.tabular_features.base_feature import BaseFeatureEncoder

folder = Path('../data/movielens/train')
csvs = [folder / c for c in os.listdir(folder)]
print(csvs)
df = pd.read_csv(csvs[0])


class VarLenFeaturesEncoder:
    def __init__(self, varlen_col):
        self.varlen_col = varlen_col
        self.max_length = None
        self.new_cols = None
        self.encoder = None

    def fit(self, df):
        self.max_length = df[self.varlen_col].apply(lambda x: len(x.split("|"))).max()
        self.new_cols = [f"{self.varlen_col}_{i}" for i in range(self.max_length)]
        df[f"{self.varlen_col}_len"] = df[self.varlen_col].apply(lambda x: len(x.split("|")))
        df[self.new_cols] = df[self.varlen_col].apply(lambda x: pd.Series(x.split("|")))
        df[self.new_cols] = df[self.new_cols].fillna("<UNK>")
        total_series = pd.concat([df[col] for col in self.new_cols])
        self.encoder = LabelEncoder()
        self.encoder.fit(total_series)

    def transform(self, df):
        df[f"{self.varlen_col}_len"] = df[self.varlen_col].apply(lambda x: len(x.split("|")))
        df[self.new_cols] = df[self.varlen_col].apply(lambda x: pd.Series(x.split("|")))
        df[self.new_cols] = df[self.new_cols].fillna("<UNK>")
        for col in self.new_cols:
            df[col] = self.encoder.transform(df[col])
        return df


class FeatureEncoder(BaseFeatureEncoder):
    def __init__(self, sparse_cols, dense_cols, varlen_cols, target_cols):
        super().__init__(sparse_cols, dense_cols, varlen_cols, target_cols)

    def fit(self, df):
        for col in self.sparse_cols:
            lbe = LabelEncoder()
            lbe.fit_transform(df[col])
            self._sparse_transformers[col] = lbe
        for col in self.dense_cols:
            mms = MinMaxScaler()
            mms.fit_transform(df[col].values.reshape(-1, 1))
            self._dense_transformers[col] = mms
        var_encoder = VarLenFeaturesEncoder(self.varlen_cols[0])
        var_encoder.fit(df)
        self._varlen_transformers[self.varlen_cols[0]] = var_encoder

    def preprocess(self, df):
        df[self.target_cols[0]] = df[self.target_cols[0] > 3].astype(int)
        return df


mvlen_sparse_cols = ['user_id', 'movie_id', 'gender', 'occupation', 'zip']
mvlen_dense_cols = ['age']
mvlen_var_cols = ['genres']
mvlen_target_cols = ['rating']

featureEncoder = FeatureEncoder(mvlen_sparse_cols, mvlen_dense_cols, mvlen_var_cols, mvlen_target_cols)
featureEncoder.fit(df)
df = featureEncoder.encode(df)
df = featureEncoder.preprocess(df)
print(df)