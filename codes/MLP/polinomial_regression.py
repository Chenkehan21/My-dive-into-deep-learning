import sys
sys.path.append('../')

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
from torch import optim
import torch.nn as nn

from utils.utils import data_iter


'''
Because of randomization the results of each experiment are different.
Sometimes even let the linear layer train 50 weights it won't overfit!
'''
 

class Net(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False), # only train weights!
        )

    def forward(self, x):
        return self.net(x)


def generate_data(max_degree=50, train_size=100, test_size=100, weights=None):
    coeff = np.zeros(max_degree)
    if weights:
        coeff[:len(weights)] = np.array(weights)
    else:
        coeff[:4] = np.array([5, 1.2, -3.4, 5.6])
    features = np.random.normal(size=(train_size + test_size, 1))
    np.random.shuffle(features)
    power = np.arange(max_degree)
    poly_features = np.power(features, power.reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)
    labels = np.dot(poly_features, coeff)
    labels += np.random.normal(scale=0.1, size=labels.shape) # add bias in dataset

    return poly_features, labels.reshape(-1, 1)


def init_weight(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0, std=0.1)


def main(lr=0.01, batch_size=8, epochs=2000):    
    # num = 4 # well fit
    num = 50 # overfit
    # num = 2 # underfit
    features, labels = generate_data()
    net = Net(input_size=num, output_size=1)
    net.apply(init_weight)
    optimizer = optim.SGD(net.parameters(), lr)
    loss_function = nn.MSELoss()

    train_data, train_labels = features[:100, :num], labels[:100]
    val_data, val_labels = features[100:, :num], labels[100:]
    train_data, train_labels, val_data, val_labels = [
        torch.tensor(x, dtype=torch.float32) for x in 
        [train_data, train_labels, val_data, val_labels]
    ]

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = 0
        net.train()
        for x, y in data_iter(batch_size, train_data, train_labels):
            loss = loss_function(net(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss

        net.eval()
        val_loss = 0
        for x, y in data_iter(batch_size, val_data, val_labels):
            loss = loss_function(net(x), y)
            val_loss += loss

        print("epoch: %d|train loss: %.3f |val_loss: %.3f"%(epoch + 1, train_loss, val_loss))
        train_losses.append(train_loss.detach().numpy())
        val_losses.append(val_loss.detach().numpy())

    print(net.net[0].weight.data)
    
    start = 200 # show overfit
    # start = 50 # show well fit and underfit
    steps = list(range(len(train_losses[start:])))
    plt.plot(steps, train_losses[start:], label='train_loss')
    plt.plot(steps, val_losses[start:], label='val_loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./polinomial_regression_overfit.png')    

if __name__ == "__main__":
    main()