import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def shallow(inp, p10, p11, p20, p21, p30, p31, q0, q1, q2, q3):
    h1 = relu(p10 + p11 * inp)
    h2 = relu(p20 + p21 * inp)
    h3 = relu(p30 + p31 * inp)
    y = q0 + q1 * h1 + q2 * h2 + q3 * h3
    return y


x = np.arange(-1, 1, 0.01)

# hidden1
y = shallow(x, 0.3, 0.9, 0.6, 0.3, -0.5, 3, 0, -0.5, 0.3, 0.2)

# hidden2
y_ = shallow(y, -0.3, -0.8, -0.7, -0.3, 0.7, -0.8, 0, 0.2, -0.4, -0.5)

# plt.plot(x, y)
plt.plot(x, y)
plt.show()

