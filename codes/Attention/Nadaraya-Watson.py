from os import pread
from d2l.torch import predict_ch3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def show_heatmaps():
    ...


def generate_data(n_train, n_test, plot=False):
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    x_test, _ = torch.sort(torch.rand(n_test) * 5)
    y_true = 2 * torch.sin(x_train) + x_train**0.8
    epsilon = torch.normal(mean=0, std=0.3, size=y_true.shape)
    y_train = y_true + epsilon

    if plot:
        plt.plot(x_train, y_true, label='true', linestyle='-', color='r')
        plt.scatter(x_train, y_train, label='with_noise')
        plt.legend()
        plt.grid(True)
        plt.savefig('./images/Nadaraya-Watson_data.png')
        plt.cla()

    return x_train, y_train, y_true, x_test


def avg_predict():
    x_train, y_train, y_true, x_test = generate_data(50, 50, True)
    y_hat = torch.repeat_interleave(y_train.mean(), y_train.numel())
    plt.plot(x_train, y_true, label='true', linestyle='-', color='r')
    plt.scatter(x_train, y_train, label='with_noise')
    plt.plot(x_test, y_hat, label='predict', linestyle=':', color='g')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/avg_predict.png')


def Nadaraya_Watson_predict():
    x_train, y_train, y_true, x_test = generate_data(50, 50, False)
    n_test = x_test.numel()
    x_tile = x_test.repeat_interleave(n_test).reshape(-1, n_test)
    attention_weight = torch.softmax(-0.5 * (x_tile - x_train)**2, dim=1)
    print(attention_weight.shape)
    plt.imshow(attention_weight, cmap='Reds')
    plt.xlabel('queries')
    plt.ylabel('keys')
    plt.colorbar()
    plt.savefig('./images/Nadaraya-Watson_attention_weight.png')
    plt.close()
    y_hat = torch.matmul(attention_weight, y_train)
    plt.plot(x_train, y_true, label='true', linestyle='-', color='r')
    plt.scatter(x_train, y_train, label='with_noise')
    plt.plot(x_test, y_hat, label='predict', linestyle=':', color='g')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/Nadaraya-Watson_predict.png')


if __name__ == "__main__":
    Nadaraya_Watson_predict()
