import sys
sys.path.append('../')

import math
import torch
import torch.nn as nn

from Modern_RNNs.seq2seq import *
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding


class PositionwiseFFN(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(input_size, num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens, output_size)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


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
    def __init__(self, key_size, query_size, value_size, 
                 num_hiddens, num_heads, dropout, 
                 ffn_input, ffn_hiddens, norm_input,
                 idx, training,
                 **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        self.training = training
        self.attention1 = MultiHeadAttention(key_size=key_size, query_size=query_size, value_size=value_size, 
                                             num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout)
        self.add_norm1 = AddNorm(input_shape=norm_input, dropout=dropout)
        self.attention2 = MultiHeadAttention(key_size=key_size, query_size=query_size, value_size=value_size, 
                                             num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout)
        self.add_norm2 = AddNorm(input_shape=norm_input, dropout=dropout)
        self.ffn = PositionwiseFFN(input_size=ffn_input, output_size=num_hiddens, num_hiddens=ffn_hiddens)
        self.add_norm3 = AddNorm(input_shape=norm_input, dropout=dropout)
    
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

        return output6, state


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers,
                 key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout,
                 ffn_input, ffn_hiddens, norm_input, **kwargs):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_enc = PositionalEncoding(encoding_dim=num_hiddens, dropout=dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size,
                             num_hiddens, num_heads, dropout,
                             ffn_input, ffn_hiddens, norm_input)
            )

    def forward(self, x, valid_lens, *args):
        x = self.pos_enc(self.embedding(x) * math.sqrt(self.num_hiddens))
        self._attention_weights = [None] * len(valid_lens)
        for i, enc in enumerate(self.blks):
            x = enc(x, valid_lens)
            self._attention_weights[i] = enc.attention.attention_weights
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, training, vocab_size, num_layers,
                 key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout,
                 ffn_input, ffn_hiddens, norm_input, **kwargs):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_enc = PositionalEncoding(encoding_dim=num_hiddens, dropout=dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, 
                             num_hiddens, num_heads, dropout,
                             ffn_input, ffn_hiddens, norm_input,
                             i, training)
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return enc_outputs, enc_valid_lens, [None] * self.num_layers

    def forward(self, x, state):
        x = self.pos_enc(self.embedding(x) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, dec in enumerate(self.blks):
            x, state = dec(x, state)
            self._attention_weights[0][i] = dec.attention1.attention_weights
            self._attention_weights[1][i] = dec.attention2.attention_weights
        
        return self.dense(x), state

    @property
    def attention_weights(self):
        return self._attention_weights


def main(batch_size=64, epochs=300, lr=0.005, 
         num_steps=10, embedding_dim=32, num_hiddens=32, num_layers=2, dropout=0.1,
         ffn_num_input=32, ffn_num_hiddens=64, num_heads=4,
         key_size=32, query_size=32, value_size=32, norm_shape=[32],
         fig_name='transformer', to_predict=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, source_vocab, target_vocab = load_data_iter(num_steps, batch_size)

    encoder = TransformerEncoder(vocab_size=len(source_vocab), num_layers=num_layers, 
                                 key_size=key_size, query_size=query_size, value_size=value_size, 
                                 num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout, 
                                 ffn_input=ffn_num_input, ffn_hiddens=ffn_num_hiddens, norm_input=norm_shape)
    decoder = TransformerDecoder(training=True, vocab_size=len(target_vocab), num_layers=num_layers, 
                                 key_size=key_size, query_size=query_size, value_size=value_size, 
                                 num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout,
                                 ffn_input=ffn_num_input, ffn_hiddens=ffn_num_hiddens, norm_input=norm_shape)
    net = EncoderDdecoder(encoder, decoder).to(device)
    loss_fun = MaskedSoftmaxCELoss()
    optimizer = optim.SGD(net.parameters(), lr)
    
    train(epochs, data_iter, target_vocab, 
          net, loss_fun, optimizer, device, 
          fig_name=fig_name)

    if to_predict:
        engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
        for eng, fra in zip(engs, fras):
            translation, attention_weight_seq = predict(
            net, device, eng, source_vocab, target_vocab, num_steps, device)
            print(f'{eng} => {translation}, bleu={BLEU(translation, fra, k=2):.3f}')
        attention_weights = torch.cat([step[0][0][0] for step in attention_weight_seq], 0).reshape((1, 1, -1, num_steps)).squeeze()
        attention_weights = attention_weights.detach().cpu().numpy()
        plt.imshow(attention_weights, cmap='Reds')
        plt.xlabel('queries')
        plt.ylabel('keys')
        plt.colorbar()
        plt.savefig('./images/multi_attention_weight.png')
        plt.close()

if __name__ == "__main__":
    main()