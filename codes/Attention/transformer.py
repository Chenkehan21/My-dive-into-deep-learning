import sys
sys.path.append('../')

import torch
import torch.nn as nn

from Modern_RNNs.seq2seq import *
from multi_head_attention import MultiHeadAttention


class PositionwiseFFN(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(input_size, num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens, output_size)

    def forward(self, x):
        return self.dense1(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    def __init__(self, input_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_shape)
    
    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


class EncoderBlock(Encoder):
    def __init__(self, key_size, query_size, value_size, 
                 num_hiddens, num_heads, dropout, 
                 ffn_input, ffn_hiddens, norm_input,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size=key_size, query_size=query_size, value_size=value_size, 
                                            num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(input_size=ffn_input, output_size=num_hiddens, num_hiddens=ffn_hiddens)
        self.add_norm1 = AddNorm(input_shape=norm_input, dropout=dropout)
        self.add_norm2 = AddNorm(input_shape=norm_input, dropout=dropout)

    def forward(self, x, valid_lens):
        output1 = self.attention(x, x, x, valid_lens)
        output2 = self.add_norm1(x, output1)
        output3 = self.ffn(output2)
        output4 = self.add_norm2(output2, output3)

        return output4


class DecoderBlock(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        self.training = training
        self.attention1 = MultiHeadAttention(key_size=, query_size=, value_size=, 
                                             num_hiddens=, num_heads=, dropout=)
        self.add_norm1 = AddNorm(input_shape=, dropout=)
        self.attention2 = MultiHeadAttention(key_size=, query_size=, value_size=, 
                                             num_hiddens=, num_heads=, dropout=)
        self.add_norm2 = AddNorm(input_shape=, dropout=)
        self.ffn = PositionwiseFFN(input_size=, output_size=, num_hiddens=)
        self.add_norm3 = AddNorm(input_shape=, dropout=)
    
    def forward(self, x, state):
        # masked multihead attention
        enc_output, enc_valid_lens, hidden_state = state
        if hidden_state[self.idx] is None:
            key_value = x
        else:
            # usually dim 1 represents number of features
            key_value = torch.cat([hidden_state[self.idx], x], dim=1)
        hidden_state[self.idx] = key_value
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        
        output1 = self.attention1(x, key_value, key_value, dec_valid_lens)
        output2 = self.add_norm1(x, output1)
        output3 = self.attention2(output2, enc_output, enc_output, enc_valid_lens)
        output4 = self.add_norm2(output2, output3)
        output5 = self.ffn(output4)
        output6 = self.add_norm3(output4, output5)

        return output6


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