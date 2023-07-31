import numpy as np
import math
from ml.data_process import normalize


class L1_regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2_regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class L1_L2_regularization:
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)

from dataclasses import dataclass

@dataclass
class RegInfo:
    epoch: int
    reg_loss: np.ndarray
    train_loss: np.ndarray
    w: np.ndarray
    grad_w: np.ndarray
    train_mse: np.ndarray
    valid_mse: np.ndarray

class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent
    variables X.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.reg_info = []

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y, X_valid, y_valid):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])
        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):

            y_pred = X.dot(self.w)
            # Calculate l2 loss
            # Gradient of l2 loss w.r.t w
            grad_w = (-(y - y_pred).dot(X) + self.regularization.grad(self.w)) / X.shape[0]
            # Update the weights
            self.w -= self.learning_rate * grad_w
            train_mse = self.cal_mase(y, y_pred)
            valid_mse = self.cal_mase(y_valid, self.predict(X_valid))
            self.reg_info.append(RegInfo(i,
                                         np.mean(self.regularization(self.w)),
                                         train_mse,
                                         self.w,
                                         grad_w,
                                         train_mse,
                                         valid_mse))

    def cal_mase(self, y, y_pred):
        return np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y, X_valid, y_valid):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y, X_valid, y_valid)


class RegressionCheck:
    def __init__(self, model):
        self.model = model

    def r2_val(self, y, y_pred):
        y_mean = np.mean(y)
        tss = np.sum(np.square(y - y_mean))
        ess = np.sum(np.square(y_pred - y_mean))
        rss = np.sum(np.square(y - y_pred))
        r2_score = ess / tss
        return r2_score