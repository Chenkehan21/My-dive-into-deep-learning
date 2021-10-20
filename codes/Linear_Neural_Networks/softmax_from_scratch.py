import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def get_data():
    trans = transforms.ToTensor()
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=trans, download=False
    )
    fmnist_val = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=trans, download=False
    )

    print(len(fmnist_train), len(fmnist_val))


if __name__ == "__main__":
    get_data()