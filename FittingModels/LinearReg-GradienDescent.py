import torch
import matplotlib.pyplot as plt


torch.random.manual_seed(10)
x = torch.arange(0, 25, 1.5)
y = torch.tensor(list(map(lambda x: x + (torch.randn(1)*5), (5 * x + 1))))


class LogGradLinearOptim:
    def __init__(self, x, y, p1=0, p0=0, std_loss=15):
        self.x = x
        self.y = y
        self.std = std_loss
        self.p1 = torch.tensor(p1, dtype=torch.float16, requires_grad=True)
        self.p0 = torch.tensor(p0, dtype=torch.float16, requires_grad=True)

    def normal_func(self, mean, x_val):
        normal_val = (1 / (self.std * torch.sqrt(torch.tensor(2*torch.pi)))) * torch.exp(-0.5 * ((x_val - mean) / self.std)**2)
        return normal_val

    def linear_model(self, point):
        return (self.p1 * point) + self.p0

    def neg_log_loss(self):
        log_likelihood = torch.zeros(1)
        i = 0
        for point in self.x:
            mean = self.linear_model(point=point)
            log_probability = torch.log(self.normal_func(mean, self.y[i]))
            log_likelihood += log_probability
            i += 1
        return -log_likelihood

    def train(self, epochs, lr=0.01):
        for i in range(epochs):
            print("p1 = ", self.p1.item(), " p0 = ", self.p0.item(), "   loss = ", self.neg_log_loss().item())
            self.neg_log_loss().backward()

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
