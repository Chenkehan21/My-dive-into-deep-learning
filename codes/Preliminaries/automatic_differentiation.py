import torch

x = torch.arange(4, dtype=torch.float32, requires_grad=True)
y = 2 * torch.dot(x, x)
print(x)
print(y)
y.backward()
print(x.grad)
x.grad.zero_()
print(x.grad)

u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)