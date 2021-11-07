import torch
import torch.nn as nn
from torch import optim

def conv2d(x, k):
    h_in, w_in= x.shape
    h_k, w_k = k.shape
    h_out, w_out = h_in - h_k + 1, w_in - w_k + 1
    res = torch.zeros(h_out, w_out)
    for i in range(h_out):
        for j in range(w_out):
            res[i, j] = (x[i : i + h_k, j : j + w_k] * k).sum()
    
    return res


class Conv2d(nn.Module):
    def __init__(self, size:tuple) -> None:
        super().__init__()
        h, w = size
        self.weight = torch.normal(0, 0.001, (h, w), requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        res = conv2d(x, self.weight) + self.bias
        return res


def train(lr=0.5, epochs=50):
    x = torch.ones(6, 8)
    x[:, 2:6] = 0
    y = torch.zeros(6, 7)
    y[:, 1] = 1
    y[:, -2] = -1

    conv = Conv2d((1, 2))
    loss_function = nn.MSELoss()
    optimizer = optim.SGD([conv.weight, conv.bias], lr=lr)
    for epoch in range(epochs):
        loss = loss_function(conv(x), y)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        conv.zero_grad()
    print(conv.weight)


if __name__ == "__main__":
    # data = torch.randn(5, 5)
    # k = torch.randn(3, 3)
    # res = conv2d(data, k)
    # print(res, res.shape)

    # conv = Conv2d((3, 3))
    # res2 = conv(data)
    # print(res2, res.shape)

    # data2 = torch.ones(6, 8)
    # data2[:, 2:6] = 0
    # print(data2)
    # k2 = torch.tensor([[1., -1.]])
    # res3 = conv2d(data2, k2)
    # print(res3)
    # print(conv2d(data2.T, k2))

    train()