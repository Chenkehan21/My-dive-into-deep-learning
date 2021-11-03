'''
this version can hardly tain well, it's a bad version with some potential bugs:
1. need to capture variable in lambda function, otherwise the weights and bias
wil be overlapped by the last one
2. the implementation of dropout layer is not pythonic and didn't consider p = 1 and p = 0
3. the self-implemented sgd and cross entropy may perform poorly
4. the gradientable variables such as w and b are non-leaf tensor
'''

import sys
sys.path.append('../')

from collections import OrderedDict

import torch

from utils.utils import load_FMNIST, softmax, cross_entropy, sgd


def dropout_layer(layer, theta=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.rand(1).to(device)
    num = int(theta * len(layer))
    if p <= theta:
        layer[torch.randint(0, len(layer), (num,))] = 0.
        layer /= 1 - p
    
    return layer


class Net:
    def __init__(self, W, B):
        idx = 0
        self.layers = OrderedDict()
        for w, b in zip(W[:-1], B[:-1]):
            self.layers[idx] = lambda x, w=w, b=b: torch.matmul(x, w) + b # pay attension to capture variable in lambda function!!!
            idx += 1
            self.layers[idx] = dropout_layer
            idx += 1
        self.layers[idx] = lambda x: torch.matmul(x, W[-1]) + B[-1]
        idx += 1
        self.layers[idx] = softmax

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        
        return x


def train(batch_size=256, lr=0.01, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = load_FMNIST(batch_size)
    w1 = torch.normal(0, 0.01, (784, 256), dtype=torch.float32, requires_grad=True).to(device)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(device)
    w2 = torch.normal(0, 0.01, (256, 128), dtype=torch.float32, requires_grad=True).to(device)
    w3 = torch.normal(0, 0.01, (128, 10), dtype=torch.float32, requires_grad=True).to(device)
    W = [w1, w2, w3]
    B = [b] * 3

    '''
    since we didn't write back propgation, the gradientable variables
    such as w and b are non-leaf tensor whose grads can't be accessed
    so we need to retain_grad
    '''
    for item in W + B:
        item.retain_grad() 
    net = Net(W, B)

    for epoch in range(epochs):
        train_loss, val_loss, n_train, n_val = 0, 0, 0, 0
        for x, y in train_set:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(-1, 784)
            y = y.reshape(-1, 1)
            n_train += 1
            loss = cross_entropy(net.forward(x), y).sum()
            loss.backward()
            train_loss += loss
            sgd(W + B, lr, batch_size)
        print("epoch: %d |train_loss: %.3f" % (epoch + 1, train_loss / n_train))


if __name__ == "__main__":
    train()