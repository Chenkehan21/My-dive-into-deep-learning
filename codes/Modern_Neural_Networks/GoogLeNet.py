import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import load_FMNIST, train_FMNIST
from batch_normalization import BatchNorm


class Inception(nn.Module):
    def __init__(self, in_channel, out_channels) -> None:
        super().__init__()
        self.p1 = nn.Conv2d(in_channel, out_channels[0], kernel_size=1, padding=0, stride=1)
        self.b1 = BatchNorm(out_channels[0], 4)
        
        self.p2_1 = nn.Conv2d(in_channel, out_channels[1][0], kernel_size=1, padding=0, stride=1)
        self.b2_1 = BatchNorm(out_channels[1][0], 4)
        self.p2_2 = nn.Conv2d(out_channels[1][0], out_channels[1][1], kernel_size=3, padding=1, stride=1)
        self.b2_2 = BatchNorm(out_channels[1][1], 4)
        
        self.p3_1 = nn.Conv2d(in_channel, out_channels[2][0], kernel_size=1, padding=0, stride=1)
        self.b3_1 = BatchNorm(out_channels[2][0], 4)
        self.p3_2 = nn.Conv2d(out_channels[2][0], out_channels[2][1], kernel_size=5, padding=2, stride=1)
        self.b3_2 = BatchNorm(out_channels[2][1], 4)
        
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channel, out_channels[3], kernel_size=1, padding=0, stride=1)
        self.b4_2 = BatchNorm(out_channels[3], 4)
        
    def forward(self, x):
        p1 = F.relu(self.b1(self.p1(x)))
        p2 = F.relu(self.b2_2(self.p2_2(F.relu(self.b2_1(self.p2_1(x))))))
        p3 = F.relu(self.b3_2(self.p3_2(F.relu(self.b3_1(self.p3_1(x))))))
        p4 = F.relu(self.b4_2(self.p4_2((self.p4_1(x)))))
        
        return torch.cat((p1, p2, p3, p4), dim=1)
    

class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), BatchNorm(64, 4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), BatchNorm(64, 4), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), BatchNorm(192, 4), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block3 = nn.Sequential(
            Inception(192, [64, (96, 128), (16, 32), 32]),
            Inception(256, [128, (128, 192), (32, 96), 64]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block4 = nn.Sequential(
            Inception(480, [192, (96, 208), (16, 48), 64]),
            Inception(512, [160, (112, 224), (24, 64), 64]),
            Inception(512, [128, (128, 256), (24, 64), 64]),
            Inception(512, [112, (144, 288), (32, 64), 64]),
            Inception(528, [256, (160, 320), (32, 128), 128]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block5 = nn.Sequential(
            Inception(832, [256, (160, 320), (32, 128), 128]),
            Inception(832, [384, (192, 384), (48, 128), 128]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.net = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            nn.Linear(1024, 10)
        )
        
    def foreard(self, x):
        return self.net(x)
    

def main(batch_size=200, lr=0.1, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=96)
    net = GoogLeNet().net
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "GoogLeNet_BN")
    
    
if __name__ == "__main__":
    main()