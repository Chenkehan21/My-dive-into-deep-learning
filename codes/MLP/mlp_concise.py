import sys
sys.path.append('../')

from utils.utils import load_FMNIST, train_FMNIST

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0, std=0.01)


def main(batch_size=256, lr=1e-1, epochs=10, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MLP().to(device)
    net.apply(init_weights)
    train_data, val_data = load_FMNIST(batch_size, num_workers)
    train_FMNIST(net, train_data, val_data, device, lr, epochs, fig_name="mlp_concise")


if __name__ == "__main__":
    main()
    # net = MLP()
    # print(net)
    # print(net._modules)