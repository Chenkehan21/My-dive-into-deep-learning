import sys
sys.path.append('../')

from utils.utils import load_FMNIST, right_prediction, val_acc2

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def init_weights():
    w1 = torch.normal(0, 0.1, (784, 256), requires_grad=True)
    b1 = torch.zeros(256, requires_grad=True)
    w2 = torch.normal(0, 0.1, (256, 10), requires_grad=True)
    b2 = torch.zeros(10, requires_grad=True)

    return [w1, w2], [b1, b2]


def ReLU(x):
    z = torch.zeros_like(x)
    return torch.max(x, z)


def mlp(x, w:list, b:list):
    x = x.reshape(-1, 784)
    h = ReLU(torch.matmul(x, w[0]) + b[0])
    return torch.matmul(h, w[1]) + b[1]


def train(batch_size=256, lr=1e-1, epochs=10, num_workers=8):
    train_iter, val_iter = load_FMNIST(batch_size, num_workers)
    w, b = init_weights()
    net = mlp
    # here we use API, our sgd and cross_entropy may encounter many problems such as bad numerical stability
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=w + b, lr=lr)
    train_accs, val_accs, train_losses = [], [],[]

    for epoch in range(epochs):
        right_cnt, sample_cnt, total_loss, n = 0, 0, 0, 0
        for feature, label in train_iter:
            n += 1
            y_hat = net(feature, w, b)
            right_cnt += right_prediction(y_hat, label)
            sample_cnt += len(y_hat)
            loss = loss_function(y_hat, label)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        acc = val_acc2(net, val_iter, w, b)
        train_acc = right_cnt / sample_cnt
        val_accs.append(acc)
        train_accs.append(train_acc)
        train_loss = total_loss / n
        train_losses.append(train_loss)
        print("epoch: ", epoch + 1, "| train_loss: %.3f"%train_loss, 
        "| train_acc: %.3f"%train_acc, "| val_acc: %.3f"%acc)
    
    steps = list(range(epochs))
    plt.plot(steps, train_accs, label='train_acc')
    plt.plot(steps, val_accs, label='val_acc')
    plt.plot(steps, train_losses, label='train_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("./mlp_from_scratch2.png")


if __name__ == "__main__":
    train()