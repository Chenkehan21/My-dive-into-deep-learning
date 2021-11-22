import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.cuda.amp import autocast
# torch.backends.cudnn.enabled = False

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import random
from tqdm import tqdm

from Recurrent_Neural_Network.rnn_from_scratch import train

def set_axes(axes, xlabel, ylabel, 
             xlim, ylim, 
             xscale, yscale, 
             xticks, yticks,
             legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if hasattr(xticks, "__len__"):
        axes.set_xticks(xticks)
    if hasattr(yticks, "__len__"):
        axes.set_yticks(yticks)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(x, y=None,
        xlabel=None, ylabel=None, 
        xlim=None, ylim=None, 
        xscale=None, yscale=None,
        xticks=None, yticks=None,
        axes=None, legend=None,
        formats=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), name=None):
    def has_one_axis(x):
        return (hasattr(x, 'ndim') and x.ndim == 1 or 
        isinstance(x, list) and not hasattr(x[0], '__len__'))
    
    axes = axes if axes else plt.gca()
    
    if legend is None:
        legend = []
    
    if has_one_axis(x):
        x = [x]
    if y is None:
        x, y = [[]] * len(x), x
    elif has_one_axis(y):
        y = [y]
    if len(x) != len(y):
        x = x * len(y)
    
    axes.cla()
    for m, n, f in zip(x, y, formats):
        if len(m):
            axes.plot(m, n, f)
        else:
            axes.plot(n, f)
    set_axes(axes, 
             xlabel, ylabel, 
             xlim, ylim, 
             xscale, yscale, 
             xticks, yticks,
             legend)
    if name == None:
        plt.savefig('./tmp.png')
    else:
        plt.savefig('./%s.png'%name)


def data_iter(batch_size, features, labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        index_mask = indices[i : min(i + batch_size, num_samples)]
        yield features[index_mask], labels[index_mask]


def right_prediction(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cnt = y_hat.type(y.dtype) == y
    return cnt.sum().item()


def val_acc(net, val_data, device):
    net.eval()
    with torch.no_grad():
        right_cnt, sample_cnt = 0, 0
        for feature, label in val_data:
            feature, label = feature.to(device), label.to(device)
            right_cnt += right_prediction(net(feature), label) 
            sample_cnt += len(label)
    
    return right_cnt / sample_cnt


def val_acc2(net, val_iter, w, b):
    right_cnt, sample_cnt = 0, 0
    if not isinstance(net, nn.Module):
        for feature, label in val_iter:
            right_cnt += right_prediction(net(feature, w, b), label)
            sample_cnt += len(label)
    else:
        for feature, label in val_iter:
            right_cnt += right_prediction(net(feature), label)
            sample_cnt += len(label)
    
    return right_cnt / sample_cnt


def load_FMNIST(batch_size, num_workers=8, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='../FMNIST_DATASET', train=True, transform=trans, download=False
    )
    fmnist_val = torchvision.datasets.FashionMNIST(
        root='../FMNIST_DATASET', train=False, transform=trans, download=False
    )

    train_iter = data.DataLoader(fmnist_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_iter = data.DataLoader(fmnist_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_iter, val_iter  


def train_FMNIST(net, train_data, val_data, device, lr=1e-1, epochs=10, fig_name=None):
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=lr)
    train_accs, val_accs, train_losses = [], [],[]

    for epoch in range(epochs):
        net.train()
        right_cnt, sample_cnt, total_loss, n = 0, 0, 0, 0
        for feature, label in tqdm(train_data):
            n += 1
            feature, label = feature.to(device), label.to(device)
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
            y_hat = net(feature)
            right_cnt += right_prediction(y_hat, label)
            sample_cnt += len(y_hat)
            loss = loss_function(y_hat, label)
            total_loss += loss
            loss.backward()
            # print(net[0][0].weight.grad)
            optimizer.step()
            optimizer.zero_grad()
        if device == torch.device('cuda'):
            torch.cuda.empty_cache()
        acc = val_acc(net, val_data, device)
        train_acc = right_cnt / sample_cnt
        val_accs.append(acc)
        train_accs.append(train_acc)
        train_loss = total_loss / n
        train_losses.append(train_loss.detach().to("cpu").numpy())
        print("epoch: ", epoch + 1, "| train_loss: %.3f"%train_loss, 
        "| train_acc: %.3f"%train_acc, "| val_acc: %.3f"%acc)
    
    if fig_name:
        steps = list(range(epochs))
        plt.plot(steps, train_accs, label='train_acc')
        plt.plot(steps, val_accs, label='val_acc')
        plt.plot(steps, train_losses, label='train_loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("./images/%s.png" % fig_name)


def loss_func(y_hat, y):
    return (y_hat - y.reshap(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def linear_model(x, w, b):
    return torch.matmul(x, w) + b


def softmax(x):
    e_x = torch.exp(x)
    return e_x / torch.sum(e_x, dim=1, keepdim=True)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


class RNN_FROM_SCRATCH:
    def __init__(self, vocab_size, hidden_size, 
                 params, init_state_fn, forward_fn,
                 device):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params = params
        self.forward_fn = forward_fn
        self.init_state_fn = init_state_fn
        self.device = device

    def __call__(self, x_batch, state):
        x = F.one_hot(x_batch.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params, self.device)
    
    def hidden_state_init(self, batch_size):
        return self.init_state_fn(batch_size, self.hidden_size, self.device)


class RNN_Moduel(nn.Module):
    def __init__(self, forward_fn, init_state_fn, vocab_size, hidden_size, device):
        super().__init__()
        self.forward_fn = forward_fn
        self.init_state_fn = init_state_fn
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        
        if self.forward_fn.bidirectional:
            self.num_directions = 2
            self.linear = nn.Linear(hidden_size * 2, self.vocab_size)
        else:
            self.num_directions = 1
            self.linear = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, x, hidden_state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32).to(self.device)
        y, hidden_state = self.forward_fn(x, hidden_state) # y.shape=(num_steps, batch_size, hidden_size)
        y = self.linear(y.reshape(-1, y.shape[-1]))
        
        return y, hidden_state
    
    def hidden_state_init(self, batch_size):
        return self.init_state_fn(self.num_directions, self.forward_fn.num_layers, 
                                  batch_size, self.hidden_size, self.device)
    

def train_rnn():
    return train        