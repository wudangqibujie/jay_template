import numpy
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from copy import deepcopy


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


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

def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


class CrossEntropy:
    def __call__(self, y, pred_y):
        val1 = np.sum(np.log(pred_y[y == 1])) / y.shape[0]
        val2 = np.sum(np.log((1 - pred_y[y == 0]))) / y.shape[0]
        return -(val1 + val2)


@dataclass
class LogInfo:
    epoch: int
    param: numpy.array
    gradients: numpy.array
    train_loss: float
    valid_loss: float

class LogisticRegression:
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    """
    def __init__(self, learning_rate=.1, gradient_descent=True, regularization=None):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()
        self.regularization = regularization
        self.loss = CrossEntropy()
        self.loginfo = []
        self.params = []

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, X_valid, y_valid, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                grad = (y_pred - y).dot(X) / y.shape[0]
                if self.regularization:
                    reg_grad = self.regularization.grad(self.param) / y.shape[0]
                    grad += reg_grad
                self.param -= self.learning_rate * grad
                # grad = -(y - y_pred).dot(X) / y.shape[0]
                # self.param -= self.learning_rate * grad
                # if i % 100 == 0:
                print(f'train_loss: {round(self.loss(y, y_pred), 5)} | valid_loss: {round(self.loss(y_valid, self.sigmoid(X_valid.dot(self.param))), 5)} | auc: {round(roc_auc_score(y_valid, 1 - self.sigmoid(X_valid.dot(self.param))), 5)}')
                self.params.append(np.expand_dims(deepcopy(self.param), axis=1))
                self.loginfo.append(LogInfo(i, self.param, grad, self.loss(y, y_pred), self.loss(y_valid, self.sigmoid(X_valid.dot(self.param)))))
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def predict_proba(self, X):
        y_pred = self.sigmoid(X.dot(self.param))
        return y_pred

