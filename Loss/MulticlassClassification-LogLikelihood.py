import numpy as np


def softmax(cat, values):
    sig_exp = np.sum(np.exp(values))
    return np.exp(values[cat])/sig_exp


def minimum_likelihood_criterion(inputs, cat=0):
    loss = 0
    for values in inputs:
        log_probability = np.log(softmax(cat, values))
        loss += log_probability
    return -loss


test = np.array([[12, 34, 2], [0, 90, 230], [123, 321, 231]])

print(minimum_likelihood_criterion(inputs=test, cat=0))
