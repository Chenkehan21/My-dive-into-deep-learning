import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(256, 256)
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            self.shared,
            nn.ReLU(),
            self.shared,
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def forward(self, x):
        return self.net(x)


def block():
    res = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
    )

    return res


def my_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.weight.data += torch.normal(0, 0.001, layer.weight.shape, requires_grad=True)
        layer.weight.data *= layer.weight.data.abs() >= 0.02


def main():
    net1 = Net()
    net2 = nn.Sequential()
    for i in range(3):
        net2.add_module("block%d"%i, block())
    net2.apply(my_init)
    net3 = nn.Sequential(
        net1,
        net2, 
        nn.Linear(128, 10),
        nn.Softmax(dim=1)
    )
    print(net3)
    data = torch.normal(0, 0.01, (16, 784))
    res = net3(data)
    print(res.sum(dim=1, keepdim=True))

    print(net3[1]._modules["block1"][0].bias.data)
    print(*[(name, param) for name, param in net3.named_parameters()])

    # shared parameters
    print(net3[0].net[2].weight.data.equal(net3[0].net[4].weight.data))
    net3[0].net[2].weight.data[:, 2:10] = 1.02
    print(net3[0].net[2].weight.data.equal(net3[0].net[4].weight.data))

if __name__ == "__main__":
    main()