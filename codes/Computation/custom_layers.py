import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = torch.normal(0, 0.001, (input_size, output_size))
        self.bias = torch.randn((output_size,))
    
    def forward(self, x):
        x = torch.matmul(x, self.weight) + self.bias
        return F.relu(x)

def main():
    net = nn.Sequential(
        MyLinear(784, 256),
        MyLinear(256, 128),
        MyLinear(128, 10),
        nn.Softmax(dim=1)
    )

    print(net)
    data = torch.randn(16, 784)
    res = net(data)
    print(res.sum(dim=1, keepdim=True))


if __name__ == "__main__":
    main()