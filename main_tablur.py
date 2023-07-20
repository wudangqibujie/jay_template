from dataclasses import dataclass
import json
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder


@dataclass
class SparseFeat:
    name: str
    type: str
    vocab_size: int
    embedding_dim: int = 16


@dataclass
class DenseFeat:
    name: str
    dimension: int = 1


@dataclass
class Label:
    name: str
    type: str


@dataclass
class SequenceFeat:
    name: str
    type: str


class FeatureLabelInfo:
    def __init__(self, features, labels, control_columns=None):
        self.features = features
        self.labels = labels
        self.control_columns = control_columns

    def add_feature(self, feature):
        self.features[feature.name] = feature

    def drop_features(self, features):
        for feature in features:
            self.features.pop(feature)

    @property
    def label_names(self):
        return [label["name"] for label in self.labels]

    @property
    def feature_names(self):
        return list(self.features.keys())

    @property
    def dense_feature_names(self):
        return [feature.name for feature in self.features.values() if isinstance(feature, DenseFeat)]

    @property
    def sparse_feature_names(self):
        return [feature.name for feature in self.features.values() if isinstance(feature, SparseFeat)]


    @classmethod
    def from_config(cls, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
        control_columns = config['control_columns']
        label_columns = config['label_columns']
        dense_features = config['dense_features']
        sparse_features = config['sparse_features']
        features = OrderedDict()
        for dense_feature in dense_features:
            features[dense_feature['name']] = DenseFeat(**dense_feature)
        for sparse_feature in sparse_features:
            features[sparse_feature['name']] = SparseFeat(**sparse_feature, embedding_dim=32)
        return cls(features, label_columns, control_columns)


class PandasDataSet:
    def __init__(self, df, feature_encoder):
        self.feature_encoder = feature_encoder
        self.df = df

    def __len__(self):
        return len(self.df)

    def process(self):
        self.df = self.feature_encoder.dense_feat_process(self.df)
        self.df = self.feature_encoder.sparse_feat_process(self.df)
        self.df = self.feature_encoder.label_process(self.df)

    def get_data(self, df):
        X = df[self.feature_encoder.feature_label_info.feature_names]
        y = df[self.feature_encoder.feature_label_info.label_names]
        return X, y

    @classmethod
    def load_csv(cls, csv_file, feature_encoder):
        df = pd.read_csv(csv_file)
        return cls(df, feature_encoder)


class FeatureEncoder:
    def __init__(self, feature_label_info):
        self.feature_label_info = feature_label_info

    def dense_feat_process(self, df):
        scaler = MinMaxScaler()
        df[self.feature_label_info.dense_feature_names] = scaler.fit_transform(df[self.feature_label_info.dense_feature_names])
        return df

    def sparse_feat_process(self, df):
        #fillna
        df[self.feature_label_info.sparse_feature_names] = df[self.feature_label_info.sparse_feature_names].fillna('-1', )
        for sparse_feature in self.feature_label_info.sparse_feature_names:
            df_onehot = pd.get_dummies(df[sparse_feature], prefix=sparse_feature)
            df = pd.concat([df, df_onehot], axis=1)
            df.drop(sparse_feature, axis=1, inplace=True)
            self.feature_label_info.drop_features([sparse_feature])
            for col in df_onehot.columns:
                self.feature_label_info.add_feature(DenseFeat(col, 1))
        return df

    def label_process(self, df):
        return df


if __name__ == '__main__':
    featureLabelInfo = FeatureLabelInfo.from_config('config/tablur_config/dataset_config.json')
    featureLabelInfo.drop_features(['campaign_id', 'customer'])
    featureEncoder = FeatureEncoder(featureLabelInfo)
    # train_dataset = PandasDataSet.load_csv('mini_data/tabular_csv/train_sample.csv', featureEncoder)
    # valid_dataset = PandasDataSet.load_csv('mini_data/tabular_csv/valid_sample.csv', featureEncoder)
    # test_dataset = PandasDataSet.load_csv('mini_data/tabular_csv/test_sample.csv', featureEncoder)

    train_df = pd.read_csv('mini_data/tabular_csv/train_sample.csv')
    valid_df = pd.read_csv('mini_data/tabular_csv/valid_sample.csv')

    train_df["tag"] = "train"
    valid_df["tag"] = "valid"
    df = pd.concat([train_df, valid_df], axis=0)

    dataset = PandasDataSet(df, featureEncoder)
    dataset.process()
    train_X, train_y = dataset.get_data(dataset.df[dataset.df["tag"] == "train"])
    valid_X, valid_y = dataset.get_data(dataset.df[dataset.df["tag"] == "valid"])
    print(train_X.shape, train_y.shape)
    print(valid_X.shape, valid_y.shape)


















# import pandas as pd
# import json
# df = pd.read_csv('mini_data/tabular_csv/train_sample.csv')
# dense_features = ['price']
# sparse_features = ['adgroup_id', 'pid', 'cate_id',
#        'campaign_id', 'customer', 'brand', 'cms_segid',
#        'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
#        'shopping_level', 'occupation', 'new_user_class_level']
#
# sequence_features = ['clk_seq']
# labels = ['clk']
# control_columns = ['time_stamp', 'userid']
# sparse_infos = []
# for sparse_feature in sparse_features:
#     df_sp = df[sparse_feature].nunique()
#     print(sparse_feature, df_sp)
#     sparse_infos.append({'name': sparse_feature, 'type': 'category', 'vocab_size': df_sp})
# config = {
#     "control_columns": control_columns,
#     "label_columns": [{'name': 'clk', 'type': 'binary'}],
#     "dense_features": [{'name': 'price'}],
#     "sparse_features": sparse_infos,
# }
# print(config)
# json.dump(config, open('config/tablur_config/dataset_config.json', 'w'), indent=4)