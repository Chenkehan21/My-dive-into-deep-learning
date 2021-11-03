import torch
import torch.nn as nn


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, block in enumerate(args):
            self._modules[idx] = block
    
    def forward(self, x):
        for block in self._modules.values():
            x = block(x)

        return x


class FixHiddenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 256)
        self.fix_layer = torch.normal(0, 0.01, (256, 128), requires_grad=False)
        self.lin2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = torch.matmul(x, self.fix_layer) + 0.005
        res = self.softmax(self.lin2(self.relu(x)))

        return res


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(self.block1(x)))


def mysequential():
    net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(), 
        nn.Dropout(0.5), 
        nn.Linear(256, 128),
        nn.ReLU(), 
        nn.Linear(128, 10),
        nn.Softmax(dim=1)
    )
    data = torch.normal(0, 0.01, (16, 784))
    res = net(data)
    print(res)
    print(res.sum(dim=1, keepdim=True))
    print(type(net._modules))
    print(net._modules.keys())
    print(net._modules.values())


def fix_hidden_layer():
    net = FixHiddenLayer()
    data = torch.normal(0, 0.01, (16, 784))
    res = net(data)
    print(res)


def nest_mlp():
    net = NestMLP()
    data = torch.normal(0, 0.01, (16, 784))
    res = net(data)
    print(res)    

if __name__ == "__main__":
    mysequential()
    fix_hidden_layer()
    nest_mlp()