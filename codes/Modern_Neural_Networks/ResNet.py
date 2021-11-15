import sys
sys.path.append('../')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import load_FMNIST, train_FMNIST


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1) -> None:
        super().__init__()
        self.use_1x1conv = use_1x1conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, stride=1)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=stride)
        self.relu = nn.ReLU(),

        '''
        a very import point! bn1 and bn2 actually have different parameters to learn
        so it's wrong to use just one bn! Otherwise it will overfit!!!!
        '''
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        if self.use_1x1conv:
            x = self.conv_1x1(x)
        y += x
        y = F.relu(y)
        
        return y
    

def resdiual_block(in_channels, out_channels, first_block=False, num_blocks=2):
    block = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            block.append(Residual(in_channels, out_channels, True, 2))
        else:
            block.append(Residual(out_channels, out_channels))
    
    return block
    

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block2 = nn.Sequential(*resdiual_block(64, 64, True, 2))
        self.block3 = nn.Sequential(*resdiual_block(64, 128, False, 2))
        self.block4 = nn.Sequential(*resdiual_block(128, 256, False, 2))
        self.block5 = nn.Sequential(*resdiual_block(256, 512, False, 2))
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.Linear(512, 10)
        )
        
        self.net = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4, 
            self.block5,
            self.block6,
        )
        
    def forward(self, x):
        return self.net(x)
    

def main(batch_size=256, lr=0.05, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=96)
    net = ResNet().net
    print(net)
    # exit()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "ResNet")
    

if __name__ == '__main__':
    main()