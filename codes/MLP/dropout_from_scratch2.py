import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.utils import load_FMNIST, train_FMNIST


def dropout(x, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert theta <= 1 and theta >= 0
    if theta == 1:
        return torch.zeros(x.shape)
    elif theta == 0:
        return x
    mask = torch.normal(0, 1, x.shape) > theta
    res = mask.to(device) * x / (1. - theta)
    
    return res.to(device)


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, theta1, theta2, train=True) -> None:
        super().__init__()
        self.input_size = input_size
        self.theta1 = theta1
        self.theta2 = theta2
        self.lin1 = nn.Linear(input_size, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, hidden_size2)
        self.lin3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        # softmax is needless, it has been integrated in nn.CrossEntropy

        self.train = train

    def forward(self, x):
        H1 = self.relu(self.lin1(x.reshape(-1, self.input_size)))
        if self.train:
            H2 = dropout(H1, self.theta1)
        H3 = self.relu(self.lin2(H2))
        if self.train:
            H4 = dropout(H3, self.theta2)
        H5 = self.lin3(H4)

        return H5


def main(hidden_size1=256, hidden_size2=128, theta1=0.3, theta2=0.3,
         batch_size=128, lr=0.1, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(784, 10, hidden_size1, hidden_size2, theta1, theta2).to(device)
    train_data, val_data = load_FMNIST(batch_size)
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "dropout_from_scratch")

if __name__ == "__main__":
    main()