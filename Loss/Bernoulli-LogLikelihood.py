import numpy as np


def sigmoid(x):
    return 1/(1+(np.e**(-x)))


def bernoulli_val(l, y):
    return ((1-sigmoid(l))**(1-y)) * (sigmoid(l)**y)


def negative_log_likelihood_criterion(distribution, y=1):         # y set to True
    log_likelihood = 0
    for point in distribution:
        log_probability = np.log(bernoulli_val(point, y))
        log_likelihood += log_probability

    return -log_likelihood


test_distribution = np.array([-1.2, -0.15, 2.1, -0.23, 0.24, 2.6, -2.7, -3.1, 3.2, 0.34])

print(negative_log_likelihood_criterion(distribution=test_distribution))
