import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random

from utils.utils import data_iter
from weight_decay_from_scratch import generate_data

torch.manual_seed(0)
random.seed(10)


class Net(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.zeros_(layer.bias)


def train(dimension=500, batch_size=8, lr=0.005, epochs=20, _lambda=3, manual=True):
    train_data, trian_labels, val_data, val_labels = generate_data(dimension=dimension)
    net = Net(dimension, 1)
    net.apply(init_weights)
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=_lambda)
    loss_function = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        net.train()
        for x, y in data_iter(batch_size, train_data, trian_labels):
            loss = loss_function(net(x), y) 
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if manual:
                net.net[0].weight.data.sub_(lr * net.net[0].weight.grad / batch_size)
                net.net[0].bias.data.sub_(lr * net.net[0].bias.grad / batch_size)
                net.net[0].weight.grad.zero_()
                net.net[0].bias.grad.zero_()
            
        net.eval()
        for x, y in data_iter(batch_size, val_data, val_labels):
            loss = loss_function(net(x), y)
            val_loss += loss.sum()
        train_losses.append(train_loss.detach().numpy())
        val_losses.append(val_loss.detach().numpy())
        print("epoch: %d |train_loss: %.3f |val_loss: %.3f"%(epoch + 1, train_loss, val_loss))
    # print(net.net[0].weight.data)

    steps = list(range(len(train_losses)))
    plt.figure(figsize=(10,10))
    plt.plot(steps, train_losses, label='train_loss')
    plt.plot(steps, val_losses, label='val_loss')
    plt.grid(True)
    plt.legend()
    if manual:
        plt.savefig('./weight_decay_concise_manul_lamda%d.png'%_lambda)
    else:    
        plt.savefig('./weight_decay_concise_lamda%d.png'%_lambda)



if __name__ == "__main__":
    train()