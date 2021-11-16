import sys
sys.path.append('../')

import torch
import torch.nn as nn

from utils.utils import load_FMNIST, train_FMNIST


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )


def transition_block(in_channels, out_channels):
        return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1),
        nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )


class DenseLayer(nn.Module):
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        block = []
        for i in range(num_convs):
            block.append(conv_block(growth_rate * i + in_channels, growth_rate))
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for block in self.net:
            y = block(x)
            x = torch.cat([x, y], dim=1)
        
        return x


class DenseNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        num_channels, growth_rate = 64, 32
        conv_nums = [4, 4, 4, 4]
        block1 = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blocks = []
        for i, conv_num in enumerate(conv_nums):
            blocks.append(DenseLayer(conv_num, num_channels, growth_rate))
            num_channels += growth_rate * conv_num
            if i != len(conv_nums) - 1:
                blocks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        
        self.net = nn.Sequential(
            block1, 
            *blocks,
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 10)
            )
    
    def forward(self, x):
        return self.net(x)


def main(batch_size=256, lr=0.1, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=96)
    net = DenseNet().net
    print(net)
    # exit()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "DenseNet")
    

if __name__ == '__main__':
    main()