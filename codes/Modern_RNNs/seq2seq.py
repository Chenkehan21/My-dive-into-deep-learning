import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, *args):
        raise NotImplementedError
    

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_state(self, encode_outputs, *args):
        raise NotImplementedError
    
    def forward(self, x, state):
        raise NotImplementedError
    

class EncoderDdecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input, x, *args):
        encoder_outputs = self.encoder.forward(input, *args)
        state = self.decoder.init_state(encoder_outputs, *args)
        
        return self.decoder.forward(x, state)
    

class Seq2SeqEncodcer(Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        ...