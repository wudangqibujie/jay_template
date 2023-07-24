# import pandas as pd
# from pathlib import Path
# import numpy as np
# import math
# from sklearn import datasets
# import numpy as np
# import matplotlib.pyplot as plt
# from ml.data_process import normalize, train_test_split, accuracy_score
# from ml.logistic_regression.LR_1 import LogisticRegression
# from ml.utils import Plot
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
# import warnings
# from sklearn import tree
# import xgboost
# warnings.filterwarnings("ignore")
#
# df_valid = pd.read_csv(Path(r"D:\迅雷下载\ali_dataset\NL") / "nl_item_test_bucket_1.csv")
# df_valid = df_valid[df_valid["49"] <= 1]
# X_valid, y_valid = df_valid[[str(i) for i in range(2, 49)]].values, df_valid["49"].values
# df = next(pd.read_csv(Path(r"D:\迅雷下载\ali_dataset\NL") / "nl_item_train_bucket_1.csv", chunksize=10000 * 20))
# df = df[df["49"] <= 1]
# X, y = df[[str(i) for i in range(2, 49)]].values, df["49"].values
#
# # model = xgboost.XGBClassifier(max_depth=8, learning_rate=0.01, n_estimators=300)
# # model = tree.DecisionTreeClassifier(max_depth=10)
# model = LogisticRegression()
# model.fit(X, y)
# y_pred = model.predict_proba(X_valid)
# print(y_pred.shape)
# print(roc_auc_score(y_valid, y_pred[:, 1]))
#
# # for sample_size in range(1, 21):
# #     print(10000 * sample_size)
# #     df = next(pd.read_csv(Path(r"D:\迅雷下载\ali_dataset\NL") / "nl_item_train_bucket_1.csv", chunksize=10000 * sample_size))
# #     df = df[df["49"] <= 1]
# #     X, y = df[[str(i) for i in range(2, 49)]].values, df["49"].values
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)
# #     # clf = LogisticRegression()
# #     clf = tree.DecisionTreeClassifier(max_depth=10)
# #     clf.fit(X_train, y_train)
# #     print(roc_auc_score(y_valid, clf.predict_proba(X_valid)[:, 1]))
#
#
import pandas as pd
from pathlib import Path

csv_files = [
    "us_item_train",
    "us_item_test",
    "us_user_train",
    "us_user_test"
]

data_folders = [
    # Path(r"D:\迅雷下载\ali_dataset\NL"),
    # Path(r"D:\迅雷下载\ali_dataset\ES"),
    # Path(r"D:\迅雷下载\ali_dataset\FR"),
    Path(r"D:\迅雷下载\ali_dataset\us")
]

for data_folder in data_folders:
    for csv_file in csv_files:
        columns = [str(i) for i in range(2, 50)] if 'item' in csv_file else [str(i) for i in range(2, 34)]
        df_iter = pd.read_csv(data_folder / f'{csv_file}.csv', chunksize=5000, header=None, names=['pv-id'] + columns)
        d = dict()
        flag = 0
        df_write = pd.DataFrame()
        for df in df_iter:
            df = df[df['pv-id'] % 10 == 1]
            df_write = pd.concat([df_write, df])
            flag += 1
            if flag % 100 == 0:
                print(flag * 3200, df_write.shape, data_folder, csv_file)
        df_write.to_csv(data_folder / f'{csv_file}_bucket_1.csv', index=False)