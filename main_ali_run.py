import pandas as pd
from pathlib import Path
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from ml.data_process import normalize, train_test_split, accuracy_score
from ml.logistic_regression.LR_1 import LogisticRegression
from ml.utils import Plot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
import warnings
from sklearn import tree
import xgboost
warnings.filterwarnings("ignore")

df_valid = pd.read_csv(Path(r"../data/ali_dataset") / "es_item_test_bucket_1.csv")
df_valid = df_valid[df_valid["49"] <= 1]
X_valid, y_valid = df_valid[[str(i) for i in range(2, 49)]].values, df_valid["49"].values
df = next(pd.read_csv(Path(r"../data/ali_dataset") / "es_item_train_bucket_1.csv", chunksize=10000 * 20))
df = df[df["49"] <= 1]
X_train, y_train = df[[str(i) for i in range(2, 49)]].values, df["49"].values
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
lr = LogisticRegression(0.1)
lr.fit(X_train, y_train)
for log_info in lr.loginfos:
    print(log_info)




