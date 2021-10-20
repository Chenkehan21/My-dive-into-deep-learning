import torch
import numpy as np

x = torch.arange(3, 13)
print(x, x.shape, x.size())
x = x.reshape((2, 5)) # not in place
print(x, x.shape)

x2 = torch.zeros((2, 4, 4))
print(x2)
x2 = x2.reshape((8, -1))
print(x2)

x3 = torch.randn(2, 3, 4)
print(x3)
x3 = torch.exp(x3)
print(x3)

x4 = torch.zeros(2, 2, 4)
x5 = torch.ones(3, 2, 4)
x6 = torch.randn(1, 2, 4)
print(x4, x5, x6)
x7 = torch.cat((x4, x5, x6), dim=0)
print(x7)

x8 = torch.arange(3).reshape(1, 3)
x9 = torch.arange(4, 7).reshape(3, 1)
print(x8, x9)
x10 = x8 + x9
print(x10)

x11 = torch.zeros_like(x5)
x11[:, 1, 1:3] = -32
print(x11)

x12 = np.arange(12)
x13 = torch.tensor(x12)
x14 = x13.numpy()
print(type(x12), type(x13), type(x14))

x15 = torch.sum(x13)
print(x15)
x16 = x15.item()
print(x16, type(x16), float(x16), int(x16))