import numpy as np


def MSE(y: np.array, y_pred: np.array) -> float:
    "Calculate mean squared error"
    return np.mean((y - y_pred) ** 2)


def R2Score(y: np.array, y_pred: np.array) -> float:
    "Calculate R2 score"
    s1 = np.sum((y - y_pred) ** 2)
    m = np.sum(y_pred) / y_pred.shape[0]
    s2 = np.sum((y - m) ** 2)

    return 1 - s1 / s2


def get_variance(z_pred: np.array) -> float:
    "Calculate estimated variance"
    return np.mean(np.var(z_pred, axis=1, keepdims=True))


def get_bias(z_test: np.array, z_pred: np.array) -> float:
    "Calculate estimated bias"
    z_pred_mean = np.mean(z_pred, axis=1, keepdims=True)
    return MSE(z_test, z_pred_mean)


def mean_MSE(z_test: np.array, z_pred: np.array) -> float:
    "Calculate the mean MSE over runs"
    return np.mean(np.mean((z_test - z_pred) ** 2, axis=1, keepdims=True))
