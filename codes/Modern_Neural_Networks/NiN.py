import sys
sys.path.append('../')

import torch
import torch.nn as nn
# torch.backends.cudnn.enabled = False

from utils.utils import load_FMNIST, train_FMNIST
from batch_normalization import BatchNorm


class NiN:
    def __init__(self):
        self.net = self.nin()
        self.net_bn = self.nin_bn()
    
    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        )

        return block
    
    def nin_block_bn(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), BatchNorm(out_channels, 4), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), BatchNorm(out_channels, 4), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), BatchNorm(out_channels, 4), nn.ReLU(),
        )

        return block
    
    def nin(self):
        net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            self.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        return net
    
    def nin_bn(self):
        net = nn.Sequential(
            self.nin_block_bn(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block_bn(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block_bn(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            self.nin_block_bn(384, 10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        return net


def main(batch_size=108, lr=0.1, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=224)
    net = NiN().net_bn
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "NiN_BN")


if __name__ == "__main__":
    main()