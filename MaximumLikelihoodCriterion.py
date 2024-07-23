import numpy as np
import matplotlib.pyplot as plt


# returns cords for normal curve
def normal(mean, std):
    x = np.arange(mean-5*std, mean+5*std, ((mean+5*std) - (mean-5*std))/50)    # -5sig to +5sig
    y = np.array(list(map(lambda x: (1 / (std * 2.5066)) * np.exp((-1 / 2) * ((x - mean) / std) ** 2), x)))  # normal function
    return x, y


x = np.array([0.12, 0.15, 0.21, 0.23, 0.24, 0.26, 0.27, 0.31, 0.32, 0.34])
plt.scatter(x, np.zeros_like(x))

plt.plot(normal(0.15, 0.05)[0], normal(0.15, 5)[1])
plt.plot(normal(0.25, 0.05)[0], normal(2.5, 5)[1])
plt.show()

