import torch
import random
import matplotlib.pyplot as plt

random.seed(42)
torch.random.manual_seed(10)
lst = random.sample(list(range(-1500, 1500, 5)), 200)
x = torch.tensor(lst)/100
y = torch.tensor(list(map(lambda x: torch.sin(3 + 0.06 * 15 * x) * torch.exp(torch.tensor(-((3 + 0.06 * 15 * x) ** 2)) / 32.0) + ((torch.randn(1))/75), x)))

# plt.scatter(x, y, s=5)
# plt.show()


class LogGradLinearOptim:
    def __init__(self, x, y, p1=0, p0=0, std_loss=15):
        self.x = x
        self.y = y
        self.std = std_loss
        self.p1 = torch.tensor(p1, dtype=torch.float16, requires_grad=True)
        self.p0 = torch.tensor(p0, dtype=torch.float16, requires_grad=True)

    def gabor_model(self, point):
        return torch.tensor(lambda x: torch.sin(self.p0 + 0.06 * self.p1 * x) * torch.exp(torch.tensor(-((self.p0 + 0.06 * self.p1 * x) ** 2)) / 32.0))

    def mse_loss(self):
        error = torch.zeros(1)
        i = 0
        for point in self.x:
            y_ = self.gabor_model(point=point)
            error_val = (y_ - y)**2
            error += error_val
            i += 1
        return error/i

    def train(self, epochs, lr=0.01):
        for i in range(epochs):
            print("p1 = ", self.p1.item(), " p0 = ", self.p0.item(), "   loss = ", self.mse_loss().item())
            self.mse_loss().backward()

            with torch.no_grad():
                self.p1 -= lr * self.p1.grad
                self.p0 -= lr * self.p0.grad

            self.p1.grad.zero_()
            self.p0.grad.zero_()

        return self.p1, self.p0


l1 = LogGradLinearOptim(x=x, y=y, p1=0, p0=0)
p1_, p0_ = l1.train(epochs=25)


y_ = torch.tensor(p1_ * x + p0_, dtype=torch.float16)
plt.scatter(x, y)
plt.plot(x, y_, color="r")
plt.show()

