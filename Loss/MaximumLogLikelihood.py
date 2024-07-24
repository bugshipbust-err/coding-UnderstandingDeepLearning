import numpy as np


def normal_val(mean, std, x):
    return (1 / (std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def maximum_likelihood_criterion(mean, std, distribution):
    likelihood = 1
    for point in distribution:
        output_probability = normal_val(mean, std, point)
        likelihood *= output_probability

    return likelihood


def maximum_log_likelihood_criterion(mean, std, distribution):
    log_likelihood = 1
    for point in distribution:
        log_probability = np.log(normal_val(mean, std, point))
        log_likelihood += log_probability

    return log_likelihood


test_distribution = np.array([12, 15, 21, 23, 24, 26, 27, 31, 32, 34])

print(maximum_likelihood_criterion(15, 5, test_distribution))
print(maximum_likelihood_criterion(25, 5, test_distribution))

print(maximum_log_likelihood_criterion(15, 5, test_distribution))
print(maximum_log_likelihood_criterion(25, 5, test_distribution))
