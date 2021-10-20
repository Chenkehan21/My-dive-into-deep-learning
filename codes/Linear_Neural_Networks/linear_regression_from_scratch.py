import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from utils import utils


def generate_data(num_samples=1000, need2plot=False):
    w = torch.FloatTensor([2, -3.4])
    b = torch.FloatTensor([4.2])
    
    # torch.normal(): normal distribution on the interval [-10, 10)
    # torch.rand(): uniform distribution
    # (number of samples, feature values)
    X = torch.rand((num_samples, len(w))) * 10 - 10
    labels = torch.matmul(X, w) + b # Matrix product of two tensors. don't need to distinguish mm or mv
    epsilon = torch.normal(0, 0.01, labels.shape) # noise can't have large virance otherwise will underfit
    labels += epsilon
    labels = labels.reshape(-1, 1)
    
    if need2plot:
        plt.scatter(X[:, 1].detach().numpy(), labels.detach().numpy(), 2)
        plt.savefig('./linear_regression_data.png')

    return X, labels
    

def linear_model(X, w, b):
    return torch.matmul(X, w) + b


def weight_initialize():
    w = torch.normal(0, 1, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    return w, b


def loss_function(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(lr=1e-1 * 0.3, batch_size=16, epochs=50, num_samples=1000, need2plot=False):
    features, labels = generate_data(num_samples, need2plot)
    w, b = weight_initialize()
    net = linear_model
    loss = loss_function

    for epoch in range(epochs):
        for feature, label in utils.data_iter(batch_size, features, labels):
            output = net(feature, w, b)
            l = loss(output, label)
            # print("l: ", l)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels).mean()
            print("epoch: %d, loss: %.6f" % (epoch + 1, train_l))
    
    print("w: ", w, "b: ", b)


if __name__ == "__main__":
    train(need2plot=True)