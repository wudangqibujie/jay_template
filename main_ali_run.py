import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


train_num = valid_num = 5000
n_features = 5
df_valid = next(pd.read_csv(Path(r"../data/ali_dataset") / "es_item_test_bucket_1.csv", chunksize=train_num))
df_valid['49'] = df_valid['49'].apply(lambda x: 1 if x >= 1 else 0)
X_valid, y_valid = df_valid[[str(i) for i in range(2, n_features)]].values, df_valid["49"].values
df = next(pd.read_csv(Path(r"../data/ali_dataset") / "es_item_train_bucket_1.csv", chunksize=valid_num))
df['49'] = df['49'].apply(lambda x: 1 if x >= 1 else 0)
X_train, y_train = df[[str(i) for i in range(2, n_features)]].values, df["49"].values

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)


# # reg = LinearRegression().fit(X_train, y_train)
# # pred_y = reg.predict(X_valid)
# # mse = mean_squared_error(y_valid, pred_y)
# # print(mse)
# X, y = make_regression(n_samples=1000, n_features=10, noise=20)
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
#
# from ml.linear_regression.regression_1 import LinearRegression, L2_regularization, L1_regularization
# reg = LinearRegression(n_iterations=50, learning_rate=0.01)
# reg.regularization = L2_regularization(1)
# # reg = LinearRegression()
# # reg = Regression(100, 0.0001)
# reg.fit(X_train, y_train, X_valid, y_valid)
# pred_y = reg.predict(X_valid)
# mse = mean_squared_error(y_valid, pred_y)
# print(list(reg.w))
# print(mse)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_valid, pred_y)
# print(r2)
#
# # df = pd.DataFrame(reg.reg_info)
import seaborn as sns
import matplotlib.pyplot as plt
# sns.lineplot(x="epoch", y="train_mse", data=pd.DataFrame(reg.reg_info), label="train_mse")
# sns.lineplot(x="epoch", y="valid_mse", data=pd.DataFrame(reg.reg_info), label="valid_mse")
# plt.show()
# sns.lineplot(x=range(X_valid.shape[0]), y=y_valid, label="y_valid")
# sns.lineplot(x=range(X_valid.shape[0]), y=pred_y, label="pred_y")
# plt.show()
#
# from ml.linear_regression.regression_1 import RegressionCheck
# reg_cgeck = RegressionCheck(reg)
# print(reg_cgeck.r2_val(y_valid, pred_y))
import numpy as np
# X_train, y_train = np.random.rand(10000, 10), np.random.randint(0, 2, 10000)
# X_valid, y_valid = np.random.rand(1000, 10), np.random.randint(0, 2, 1000)

# print("LogisticRegression")
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# lr = LogisticRegression(solver='sag')
# lr.fit(X_train, y_train)
# pred_y = lr.predict_proba(X_valid)
# auc = roc_auc_score(y_valid, pred_y[:, 1])
# print(auc)
# import numpy as np
# from ml.logistic_regression.LR_1 import LogisticRegression, L2_regularization
# lr1 = LogisticRegression(0.05, regularization=L2_regularization(.01))
# lr1.fit(X_train, y_train, X_valid, y_valid, n_iterations=2000)
# pred_y = lr1.predict_proba(X_valid)
# print(roc_auc_score(y_valid, 1 - pred_y))
# print(pred_y)
# df_metric = pd.DataFrame(lr1.loginfo)

# params = np.concatenate(lr1.params, axis=1)
# print(params.shape)
# sns.heatmap(params)
# plt.show()
# for i in range(10):
#     sns.lineplot(x=range(50), y=params[i, :])
#     plt.show()
# sns.lineplot(x="epoch", y="train_loss", data=df_metric, label="train_loss")
# sns.lineplot(x="epoch", y="valid_loss", data=df_metric, label="valid_loss")
# plt.show()
# df_desen = pd.DataFrame()
# df_desen["y"] = y_valid
# df_desen["pred_y"] = 1 - pred_y
# df_desen = df_desen.sort_values(by="pred_y")
# sns.lineplot(x=range(df_desen.shape[0]), y="y", data=df_desen, label="y")
# plt.show()

from ml.tree.tree_1 import ClassificationTree, RegressionTree
from sklearn.metrics import roc_auc_score
tree = RegressionTree(min_samples_split=50)
tree.fit(X_train, y_train)
pred_y = tree.predict(X_valid)
print(roc_auc_score(y_valid, pred_y))


