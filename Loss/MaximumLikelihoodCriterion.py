import numpy as np
import matplotlib.pyplot as plt


def normal_val(mean, std, x):
    return (1 / (std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


# returns cords for normal curve
def normal(mean, std):
    x = np.arange(mean-5*std, mean+5*std, ((mean+5*std) - (mean-5*std))/50)    # -5sig to +5sig
    y = np.array(list(map(lambda x: (1 / (std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2), x)))
    return x, y


test_distribution = np.array([12, 15, 21, 23, 24, 26, 27, 31, 32, 34])
plt.scatter(test_distribution, np.zeros_like(test_distribution))

plt.plot(normal(15, 5)[0], normal(15, 5)[1])
plt.plot(normal(25, 5)[0], normal(25, 5)[1])
plt.show()


def maximum_likelihood_criterion(mean, std, distribution):
    likelihood = 1
    for point in distribution:
        output_probability = normal_val(mean, std, point)
        # print(output_probability)       # probability values can go beyond 1 if the std is less than 1(refer ending)
        likelihood *= output_probability

    return likelihood


print(maximum_likelihood_criterion(15, 5, test_distribution))
print(maximum_likelihood_criterion(25, 5, test_distribution))

"""
When the standard deviation(std) is less than 1, the values at 2 standard deviations (2σ) and 3 standard deviations (3σ)
away from the mean will decrease compared to the mean. Conversely, when the standard deviation is greater than 1, these
values spread out more, and the density becomes lower at those points.
"""
