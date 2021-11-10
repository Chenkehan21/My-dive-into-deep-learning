import sys
sys.path.append('../')

import torch
import torch.nn as nn

from utils.utils import load_FMNIST, train_FMNIST


class AlexNet(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, padding=1, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

        shape = self.conv_forward()
        self.linear = nn.Sequential(
            nn.Linear(shape, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def conv_forward(self):
        tmp = torch.randn(self.input_shape)
        res = self.conv(tmp)

        return res.numel()

    def forward(self, x):
        return self.linear(self.conv(x))


def main(batch_size=128, lr=0.01, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=224)
    net = AlexNet((1, 1, 224, 224))
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "AlexNet")


if __name__ == "__main__":
    main()