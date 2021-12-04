import sys
sys.path.append('../')

import torch
import torch.nn as nn

from Modern_RNNs.seq2seq import *


class PositionwiseFFN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ...


class AddNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ...


class EncoderBlock(Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, *args):
        ...


class DecoderBlock(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, state):
        ...


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    

    def forward(self, x):
        ...


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def init_state(self, x):
        ...

    def forward(self, x):
        ...


def main():
    ...


if __name__ == "__main__":
    main()