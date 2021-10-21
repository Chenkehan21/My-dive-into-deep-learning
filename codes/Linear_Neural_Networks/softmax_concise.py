from numpy.lib.arraysetops import isin
from softmax_from_scratch import load_data, right_prediction
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


'''
The nn.CrossEntropy() does LogSoftmax, so our net has no need to
build a softmax layer!

If we do built a softmax layer, the network still works, but we 
need a larger learning rate and more epochs.

LogSoftmax is a trick to avoid overflow.
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10)
        )
    
    def forward(self, x):
        return self.net(x)


def init_weight(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=0.01)


def val_acc(net, val_data, device):
    right_cnt, sample_cnt = 0, 0
    for feature, label in val_data:
        feature, label = feature.to(device), label.to(device)
        right_cnt += right_prediction(net(feature), label) 
        sample_cnt += len(label)
    
    return right_cnt / sample_cnt


def train(batch_size=256, lr=0.1, epochs=10, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_data(batch_size, num_workers)
    net = Net().to(device)
    net.apply(init_weight) # initialize weight manually
    print(next(net.parameters()).device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=lr)
    train_accs, val_accs, train_losses = [], [],[]

    for epoch in range(epochs):
        right_cnt, sample_cnt, total_loss, n = 0, 0, 0, 0
        for feature, label in train_data:
            n += 1
            feature, label = feature.to(device), label.to(device)
            y_hat = net(feature)
            right_cnt += right_prediction(y_hat, label)
            sample_cnt += len(y_hat)
            loss = loss_function(y_hat, label)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = val_acc(net, val_data, device)
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
    plt.savefig("./softmax_concise.png")

if __name__ == "__main__":
    train()