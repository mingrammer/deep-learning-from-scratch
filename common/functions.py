import numpy as np


###
# Activation functions
###
def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    x = x.astype(np.float128)
    return 1.0 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

###
# Loss functions
###
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
