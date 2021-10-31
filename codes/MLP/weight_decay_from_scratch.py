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

torch.manual_seed(0)
random.seed(10)


def init_params(input_size, output_size):
    w = torch.normal(0, 1, size=(input_size, output_size), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    return w, b


def generate_data(train_size=100, test_size=100, dimension=10, batch_size=8):
    x = torch.randn((train_size + test_size, dimension), dtype=torch.float32)
    y = (x * 0.25).sum(dim=1, keepdims=True) + 0.05
    y += torch.normal(mean=0, std=0.01, size=y.shape)

    train_data, trian_labels = x[:train_size], y[:train_size]
    val_data, val_labels = x[train_size:], y[train_size:]


    return train_data, trian_labels, val_data, val_labels


def loss_func(y_hat, y):
    return (y_hat - y)**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(dimension=500, batch_size=8, lr=0.005, epochs=20, _lambda=3):
    train_data, trian_labels, val_data, val_labels = generate_data(dimension=dimension)

    w, b = init_params(dimension, 1)
    net = lambda X: torch.matmul(X, w) + b

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss, val_loss, n_train, n_val = 0, 0, 0, 0
        for x, y in data_iter(batch_size, train_data, trian_labels):
            n_train += 1
            loss = loss_func(net(x), y) 
            train_loss += loss.sum()
            loss += _lambda * 0.5 * torch.norm(w)**2
            loss.sum().backward()
            sgd([w, b], lr, batch_size)
            
        for x, y in data_iter(batch_size, val_data, val_labels):
            n_val += 1
            loss = loss_func(net(x), y)
            val_loss += loss.sum()
        train_losses.append(train_loss.detach().numpy())
        val_losses.append(val_loss.detach().numpy())
        print("epoch: %d |train_loss: %.3f |val_loss: %.3f"%(epoch + 1, train_loss / n_train, val_loss / n_val))
    print(net.net[0].weight.data)

    steps = list(range(len(train_losses)))
    plt.figure(figsize=(10,10))
    plt.plot(steps, train_losses, label='train_loss')
    plt.plot(steps, val_losses, label='val_loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./weight_decay_from_scratch_lamda%d.png'%_lambda)


if __name__ == "__main__":
    train()