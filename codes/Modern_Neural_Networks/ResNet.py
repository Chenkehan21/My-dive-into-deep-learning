import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import load_FMNIST, train_FMNIST


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False) -> None:
        super().__init__()
        self.use_1x1conv = use_1x1conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(),
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        y = self.bn(self.conv2(F.relu(self.bn(self.conv1(x)))))
        if self.use_1x1conv:
            x = self.conv_1x1(x)
        y += x
        y = F.relu(y)
        
        return y
    

def resdiual_block(in_channels, out_channels, use_1x1conv, num_blocks):
    block = []
    for _ in range(num_blocks):
        block.append(
            nn.Sequential(
                Residual(in_channels, out_channels, use_1x1conv)
            )
        )
    
    return block
    

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block2 = nn.Sequential(*resdiual_block(64, 64, False, 2))
        self.block3 = nn.Sequential(
            Residual(64, 64, True),
            Residual(64, 64, False)
        )
        self.block4 = nn.Sequential(
            Residual(64, 64, True),
            Residual(64, 64, False)
        )
        self.block5 = nn.Sequential(
            Residual(64, 64, True),
            Residual(64, 64, False)
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.net = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4, 
            self.block5,
            self.block6,
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.net(x)
    

def main(batch_size=64, lr=0.1, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=96)
    net = ResNet().net
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "ResNet")
    

if __name__ == '__main__':
    main()