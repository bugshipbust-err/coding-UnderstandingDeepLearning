import torch
import matplotlib.pyplot as plt

torch.random.manual_seed(10)
x = torch.arange(0, 25, 1.5)
y = torch.tensor(list(map(lambda x: x + (torch.randn(1)*5), (5 * x + 2))))


class LinearLossPlot:
    def __init__(self, x, y, std_loss=15):
        self.x = x
        self.y = y
        self.std = std_loss

    def normal_func(self, mean, x_val):
        normal_val = (1 / (self.std * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((x_val - mean) / self.std)**2)
        return normal_val

    def neg_log_loss(self, p1, p0):
        log_likelihood = torch.zeros(1)
        for i, point in enumerate(self.x):
            mean = (p1 * point) + p0
            log_probability = torch.log(self.normal_func(mean, self.y[i]))
            log_likelihood += log_probability
        return -log_likelihood

    def loss_plot(self, precision=2):
        x_vals, y_vals, loss_vals = [], [], []
        for i in range(0, precision*10):
            for j in range(0, precision*10):
                l = self.neg_log_loss(i/precision, j/precision)
                x_vals.append(i/precision)
                y_vals.append(j/precision)
                loss_vals.append(l.item())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_vals, y_vals, loss_vals, c=loss_vals, cmap='viridis')
        fig.colorbar(scatter)
        plt.show()


l1 = LinearLossPlot(x=x, y=y)
l1.loss_plot(precision=2)     # precision ==> scatter plot density
