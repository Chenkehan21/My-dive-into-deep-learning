import sys

from torch.nn.modules.loss import MSELoss
sys.path.append("..")

import torch
import torch.nn as nn
from utils.utils import data_iter
from linear_regression_from_scratch import generate_data


def train(lr=1e-2, batch_size=16, epochs=10, num_samples=1000, need2plot=False):
    features, labels = generate_data(num_samples, need2plot)
    net = nn.Sequential(nn.Linear(2, 1, bias=True))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    optimizor = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(epochs):
        for feature, label in data_iter(batch_size, features, labels):
            l = loss(net(feature), label)
            optimizor.zero_grad()
            l.backward()
            optimizor.step()
        with torch.no_grad():
            train_l = loss(net(features), labels)
            print("epoch: %d, loss: %.6f" % (epoch + 1, train_l))
    
    print("w: ", net[0].weight.data, "b: ", net[0].bias.data)
            

if __name__ == "__main__":
    train()