import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


fig = plt.figure(num=1, clear=True)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

(x1, x2) = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))

h1 = relu(0.5 + 0.3*x1 + 1.3*x2)
h2 = relu(-0.5 - 0.9*x1 + 2*x2)
h3 = relu(-0.5 + 3*x1 - 0.5*x2)

y = 0 - h1 + h2 + h3

# ax1.plot_surface(x1, x2, h1, cmap="viridis")
# ax1.plot_surface(x1, x2, h2, cmap="plasma")
# ax1.plot_surface(x1, x2, h3, cmap="cividis")
# plt.show()
ax1.plot_surface(x1, x2, y, cmap="viridis")
plt.show()

