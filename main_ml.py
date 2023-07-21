import numpy as np
import math
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from ml.data_process import normalize, train_test_split, accuracy_score
from ml.logistic_regression.LR_1 import LogisticRegression
from ml.utils import Plot
from sklearn.metrics import roc_auc_score

# --------------------------------------------- LR ------------------------------------------------------
# data = datasets.load_iris()
# X = normalize(data.data[data.target != 0])
# y = data.target[data.target != 0]
# y[y == 1] = 0
# y[y == 2] = 1
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)
# import pandas as pd
#
# df = pd.DataFrame()
# df['label'] = y_test
# clf = LogisticRegression(gradient_descent=True, learning_rate=0.01, C=0.1)
# clf.fit(X_train, y_train, n_iterations=2000)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print ("Accuracy:", accuracy)
# # Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)
# print(y_test)
# print(y_pred)
# print(clf.param)
# print(clf.predcit_prob(X_test))
# df['my_score'] = clf.predcit_prob(X_test)
# print(roc_auc_score(y_test, clf.predcit_prob(X_test)))
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(X_train,y_train)
# print ('score Scikit learn: ', clf.score(X_test,y_test))
# print(clf.predict(X_test))
# print(clf.coef_)
# print(clf.predict_proba(X_test)[:, 1])
# print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
#
# df['sk_score'] = clf.predict_proba(X_test)[:, 1]


# --------------------------------------------- DT ------------------------------------------------------
# from ml.tree.tree_1 import ClassificationTree
# data = datasets.load_iris()
# X = data.data
# y = data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# clf = ClassificationTree()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(y_pred)
# print ("Accuracy:", accuracy)

# Plot().plot_in_2d(X_test, y_pred,
#                     title="Decision Tree",
#                     accuracy=accuracy,
#                     legend_labels=data.target_names)

# ---------------------------------------------RF ------------------------------------------------------
# from ml.rf.rf_1 import RandomForest
#
# data = datasets.load_iris()
# X = data.data
# y = data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)
# clf = RandomForest(n_estimators=100)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# Plot().plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


# --------------------------------------------- XGB ------------------------------------------------------
# from ml.xgb.xgb_1 import XGBoost
# data = datasets.load_iris()
# X = data.data
# y = data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)
# clf = XGBoost()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print ("Accuracy:", accuracy)
# Plot().plot_in_2d(X_test, y_pred,
#     title="XGBoost",
# accuracy=accuracy,
# legend_labels=data.target_names)


# --------------------------------------------- gbdt ------------------------------------------------------
# from ml.gbdt.gbdt_1 import GradientBoostingClassifier
#
# data = datasets.load_iris()
# X = data.data
# y = data.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#
# clf = GradientBoostingClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
#
# print ("Accuracy:", accuracy)
#
#
# Plot().plot_in_2d(X_test, y_pred,
#                 title="Gradient Boosting",
#                 accuracy=accuracy,
#                 legend_labels=data.target_names)
