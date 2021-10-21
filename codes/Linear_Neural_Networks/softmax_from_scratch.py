from numpy.lib.polynomial import roots
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time


# git add -u
# git reset -- main/dontcheckmein.txt


def get_data():
    trans = transforms.ToTensor()
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=trans, download=False
    )
    fmnist_val = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=trans, download=False
    )

    print("traindata size: ", len(fmnist_train), 
          " | valdata size:", len(fmnist_val))

    return fmnist_train, fmnist_val


def show_img():
    fmnist_train, fmnist_val = get_data()
    
    # if only one channel, give matplotlib a 2d input
    train_sample1 = fmnist_train[0][0].permute(1, 2, 0).squeeze()
    print("image shape:", train_sample1.shape) 
    plt.imshow(train_sample1)
    plt.show()


def toy_load_data(batch_size, num_workers):
    fmnist_train, fmnist_val = get_data()
    train_iter = data.DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_iter = data.DataLoader(fmnist_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, val_iter


def test_readtime():
    train_iter, val_iter = toy_load_data(256, 8)
    t = time.time()
    for feature, label in train_iter:
        continue
    print("load train data consume %.3fs"%(time.time() - t))


def load_data(batch_size, num_workers, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=trans, download=False
    )
    fmnist_val = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=trans, download=False
    )

    train_iter = data.DataLoader(fmnist_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_iter = data.DataLoader(fmnist_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_iter, val_iter


def init_weights(shape_w, shape_b):
    w = torch.normal(0, 0.01, shape_w, requires_grad=True)
    b = torch.zeros(shape_b, requires_grad=True)

    return w, b


def linear(x, w, b):
    return torch.matmul(x, w) + b


def softmax(x):
    exp_x = torch.exp(x)
    denominator = exp_x.sum(1, keepdim=True)
    return exp_x / denominator


def net(x, w, b):
    x = x.reshape(-1, w.shape[0])
    output = linear(x, w, b)
    
    return softmax(output)


def loss_function(y, y_hat):
    return -torch.log(y_hat[range(len(y_hat)), y])


def sgd(params:list, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def right_prediction(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cnt = y_hat.type(y.dtype) == y
    return cnt.sum().item()


def val_acc(net, val_iter, w, b):
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


def train(batch_size=256, lr=1e-1, epochs=10, num_workers=8):
    train_iter, val_iter = load_data(batch_size, num_workers)
    image_shape = next(iter(train_iter))[0][0].shape
    feature_size = torch.prod(torch.tensor(image_shape))
    class_num = 10
    shape_w = (feature_size, class_num)
    shape_b = class_num
    w, b = init_weights(shape_w, shape_b)
    train_accs, val_accs, train_losses = [], [],[]

    for epoch in range(epochs):
        right_cnt, sample_cnt, total_loss = 0, 0, 0
        for feature, label in train_iter:
            y_hat = net(feature, w, b)
            right_cnt += right_prediction(y_hat, label)
            sample_cnt += len(y_hat)
            loss = loss_function(label, y_hat)
            print(y_hat)
            print(loss.sum() / loss.numel())
            exit()
            total_loss += loss
            loss.backward()
            sgd([w, b], lr, batch_size)
        acc = val_acc(net, val_iter, w, b)
        train_acc = right_cnt / sample_cnt
        val_accs.append(acc)
        train_accs.append(train_acc)
        train_loss = total_loss / sample_cnt
        train_losses.append(train_loss)
        print("epoch: ", epoch + 1, "| train_loss: %.3f"%train_loss, 
        "| train_acc: %.3f"%train_acc, "| val_acc: %.3f"%acc)
    
    steps = list(range(epochs))
    plt.plot(steps, train_accs, label='train_acc')
    plt.plot(steps, val_accs, label='val_acc')
    plt.plot(steps, train_losses, label='train_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("./softmax_from_scratch.png")

if __name__ == "__main__":
    train()