import torch
import numpy as np

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
y = torch.randn_like(x)
z = x.clone()
print(x, y, z)
res = x * z
print(res)

print(x.sum(), x.sum().shape, type(x.sum()), x.sum().item())
print(x.sum(axis=0))
print(x.sum(axis=0, keepdims=True), x.sum(axis=0, keepdims=True).shape)
print(x.sum(axis=0).unsqueeze(dim=0))
print(x.sum(axis=0, keepdims=True).squeeze(dim=0))

print(x.sum(axis=1))
print(x.sum(axis=1, keepdims=True), x.sum(axis=1, keepdims=True).shape)

print(x.mean())
print(x.mean(axis=0))
print(x.mean(axis=0, keepdims=True))
print(x.mean(axis=1))
print(x.mean(axis=1, keepdims=True))
print(x.mean(axis=[0, 1]))
print(x.mean(axis=[0, 1], keepdims=True))

sum1 = x.sum(dim=0, keepdims=True)
sum2 = x.sum(dim=1, keepdims=True)
print(x / sum1)
print(x / sum2)

print(x.cumsum(axis=0))
print(x.cumsum(axis=1))
print(x.reshape(1, -1).cumsum(axis=1)) # prefix sum

print(x.T)

# dot product: 1D and 1D
x2 = torch.arange(12, dtype=torch.float32)
x3 = torch.ones(x2.shape, dtype=torch.float32)
x3[2 : 9] = -1
print(x2)
print(x3)
print(torch.dot(x2, x3))
print(torch.sum(x2 * x3))

# matrix-vector product
x4 = torch.ones(x.T.shape[0])
print(x4)
print(x.shape, x4.shape)
print(torch.mv(x, x4))

# matrix-matrix multiplication
print(torch.mm(x,x.T))
print(x @ x.T)

print(x.norm()) # Frobenius norm
print(torch.sqrt(torch.sum(x**2)))
print(x.norm(dim=0, keepdim=True)) #L2 norm 
print(x.norm(dim=1, keepdim=True)) # L2 norm

print(x.numel(), len(x))
x5 = torch.zeros((10,2,3,4,5))
print(x5.numel(), len(x5))