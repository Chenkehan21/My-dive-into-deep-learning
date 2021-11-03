import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.utils import load_FMNIST, train_FMNIST


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
         return self.net(x)


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer, std=0.01)


def main(batch_size=128, lr=0.1, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    train_data, val_data = load_FMNIST(batch_size)
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "dropout_concise")


if __name__ == "__main__":
    main()