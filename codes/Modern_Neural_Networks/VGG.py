import sys
sys.path.append('../')

import torch
import torch.nn as nn

from utils.utils import load_FMNIST, train_FMNIST


class VGG(nn.Module):
    def __init__(self, input_shape, conv_arch) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.conv_arch = conv_arch
    
    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)
    
    def vgg(self):
        conv_layers = []
        in_channels = 1
        for num_convs, out_channels in self.conv_arch:
            conv_layers.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        conv = nn.Sequential(*conv_layers)
        tmp_data = torch.randn(self.input_shape)
        shape = conv(tmp_data).numel()
        
        return nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(shape, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )


def main(batch_size=128, lr=0.05, epochs=10):
    train_data, val_data = load_FMNIST(batch_size, resize=224)
    input_shape = (1, 1, 224, 224)
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    scale = 4
    small_conv_arch = [(pair[0], pair[1] // scale) for pair in conv_arch]
    net = VGG(input_shape, small_conv_arch).vgg()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("start train")
    train_FMNIST(net, train_data, val_data, device, lr, epochs, "VGG")


if __name__ == "__main__":
    main()