import sys
sys.path.append('../')

import torch
import torch.nn as nn

from utils.utils import load_FMNIST, train_FMNIST
from Modern_Neural_Networks.batch_normalization import BatchNorm

class LeNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            # input: (batch_size, channel=1, h=28, w=28)
            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2), BatchNorm(6, 4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)), BatchNorm(16, 4),
            nn.Sigmoid(),
            nn.AvgPool2d((2, 2), stride=2),
            nn.Flatten(),
        )

        shape = self.conv_forward(input_shape)
        self.linear = nn.Sequential(
            nn.Linear(shape, 120), BatchNorm(120, 2),
            nn.Sigmoid(),
            nn.Linear(120, 84), BatchNorm(84, 2),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def conv_forward(self, input_shape):
        # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        tmp = torch.ones(input_shape).to('cpu')
        x = self.conv(tmp)
        return x.numel()

    def forward(self, x):
        return self.linear(self.conv(x))


def main(batch_size=256, lr=2.0, epochs=10):
    train_data, val_data = load_FMNIST(batch_size)
    net = LeNet((1, 1, 28, 28))
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "LeNet_BN_lr=2.0")


if __name__ == '__main__':
    main()