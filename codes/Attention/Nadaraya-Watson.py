from os import pread
from d2l.torch import predict_ch3
import torch
import torch.nn as nn
from torch import optim
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


class NWKernelRegression(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.w = nn.Parameter(torch.randn((1,), device=device, requires_grad=True))

    def forward(self, query, keys, value):
        self.attention_weight = torch.softmax(-0.5 * (self.w * (query - keys))**2, dim=1)
        y_hat = torch.bmm(self.attention_weight.unsqueeze(1), value.unsqueeze(-1))
        '''
        res = torch.bmm(a, b) if a.shape = (b, n, m) and b.shape = (b, m, p) then res.shape = (b, n, p)
        here, attention_weight.shape = (50, 49), value.shape = (50, 49)
        attention_weight.unsqueeze(1).shape = (50, 1, 49); value.unsqueeze(-1).shape = (50, 49, 1)
        then y_hat.shape = (50, 1, 1), so y_hat.squeeze().shape = (50,)
        actually it's the same as torch.sum(attention_weight * values, dim=1).squeeze()
        '''

        return y_hat.squeeze()


def train(epochs=10, lr=0.5):
    x_train, y_train, y_true, x_test = generate_data(50, 50, False)
    n_train = x_train.numel()
    x_tail = x_train.repeat(n_train, 1)
    y_tail = y_train.repeat(n_train, 1)
    keys = x_tail[(1 - torch.eye(n_train)).type(torch.bool)].reshape(n_train, -1)
    query = x_train.repeat_interleave(keys.shape[1]).reshape(n_train, -1)
    values = y_tail[(1 - torch.eye(n_train)).type(torch.bool)].reshape(n_train, -1)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    net = NWKernelRegression(device)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr)

    total_loss = []
    for epoch in range(epochs):
        loss = loss_func(net(query, keys, values), y_train)
        total_loss.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("epoch:%d| loss: %.3f" % (epoch + 1, loss))
    
    x = list(range(len(total_loss)))
    plt.plot(x, total_loss, label='train loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/Nadarayha-Watson_parameters_loss.png')
    plt.close()

    n_test = x_test.numel()
    keys_test = x_test.repeat(n_test, 1)
    query_test = x_test.repeat_interleave(keys_test.shape[1]).reshape(-1, keys_test.shape[1])
    value_test = y_train.repeat(n_test, 1)
    y_hat = net(query_test, keys_test, value_test).detach().cpu().numpy()
    plt.plot(x_train, y_true, label='true', linestyle='-', color='r')
    plt.scatter(x_train, y_train, label='with_noise')
    plt.plot(x_test, y_hat, label='predict', linestyle=':', color='g')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/Nadaraya-Watson_parameter_predict.png')
    plt.close()

    plt.imshow(net.attention_weight.detach().cpu().numpy(), cmap='Reds')
    plt.xlabel('queries')
    plt.ylabel('keys')
    plt.colorbar()
    plt.savefig('./images/Nadaraya-Watson_parameter_attention_weight.png')

if __name__ == "__main__":
    train()
