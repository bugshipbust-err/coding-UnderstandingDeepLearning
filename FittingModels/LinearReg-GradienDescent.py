import torch
import matplotlib.pyplot as plt


torch.random.manual_seed(9)
x = torch.arange(0, 25, 1.5)
y = torch.tensor(list(map(lambda x: x + (torch.randn(1)), (3 * x + 2))))
# y = 3 * x + 2


def normal_func(mean, std, x):
    return (1 / (std * torch.sqrt(torch.tensor(2*torch.pi)))) * torch.exp(-0.5 * ((x - mean) / std) ** 2)


def linear_model(x, p1, p0):
    return (p1 * x) + p0


def neg_log_loss(p1, p0, std, x_distribution, y):
    log_likelihood = 0
    i = 0
    for point in x_distribution:
        mean = linear_model(x=point, p1=p1, p0=p0)
        log_probability = torch.log(normal_func(mean, std, point))
        print(mean, y[i], " --> ", log_probability)
        log_likelihood += log_probability
        i += 1
    return -log_likelihood


print(y)
print(neg_log_loss(p1=3, p0=2, std=1, x_distribution=x, y=y))

plt.scatter(x, y)
plt.show()
