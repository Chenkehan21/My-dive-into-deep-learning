import sys
sys.path.append('../')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.utils import data_iter

def generate_data(plot=False):
    x = torch.arange(1, 1001) * 0.01
    y = torch.sin(x) + torch.normal(mean=0, std=0.2, size=(1000,))
    if plot:
        plt.plot(x.detach().to('cpu').numpy(), y.detach().to('cpu').numpy())
        plt.savefig('./images/sequence_model_data.png')

    tau = 4
    features = torch.zeros(1000 - tau, tau)
    for i in range(tau):
        features[:, i] = y[i: 1000 + i - tau]
    labels = y[tau:].reshape(-1, 1)

    return x, y, features, labels


def weight_init(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=0.01)


def train(batch_size=16, lr=0.01, epochs=10):
    raw_x, raw_y, features, labels = generate_data()
    train_features, train_labels = features[:600], labels[:600]
    val_features, val_labels = features[600:], labels[600:]
    net = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    net.apply(weight_init)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        net.train()
        total_loss, train_n = 0, 0
        for x, y in data_iter(batch_size, train_features, train_labels):
            train_n += 1
            loss = loss_func(net(x), y)
            loss.backward()
            total_loss += loss
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_loss / train_n
        print("epoch: %d| train loss: %.3f" % (epoch + 1, train_loss))
    
    # onestep_predict
    onestep_predicts = net(features).detach().to('cpu').numpy()
    plt.plot(raw_x[4:], raw_y[4:], label='raw data', linestyle='-', color='c')
    plt.plot(raw_x[4:], onestep_predicts, label='onestep_predict', linestyle=':', color='m')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/onestep_predict.png')
    plt.cla()

    # multisteps_predict
    max_step = 64
    tau = 4
    T = 1000
    new_features = torch.zeros(T - max_step - tau + 1, tau + max_step)
    for i in range(tau):
        new_features[:, i] = raw_y[i : i + T - max_step - tau + 1]
    for i in range(tau, tau + max_step):
        new_features[:, i] = net(new_features[:, i - tau : i]).reshape(-1)

    new_features = new_features.detach().to('cpu').numpy()
    steps = (1, 4, 16, 64)
    plt.plot(raw_x[:T - max_step - tau + 1:], raw_y[:T - max_step - tau + 1],
             label='raw data')
    for step in steps:
        plt.plot(raw_x[:T - max_step - tau + 1], new_features[:, tau + step - 1],
                 label='%dsteps predict'%step)
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/multisteps_predict.png')

        

if __name__ == "__main__":
    train()