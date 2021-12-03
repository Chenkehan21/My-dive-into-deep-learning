import sys
sys.path.append('../')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from attention_scoring_functions import DotProductAttention
from Modern_RNNs.seq2seq import *
from bahdanau_attention import Seq2SeqAttentionDecoder


def transpose_qkv(x, num_heads):
    # x.shape=(batch_size, num_queries, num_hiddens)
    # we want output.shape=(batch_size * num_heads, num_queries, num_hiddens / num_heads)
    x = x.reshape(x.shape[0] * num_heads, x.shape[1], -1)

    return x



def transpose_output(x, num_heads):
    # x.shape=(batch_size * num_heads, num_queries, num_hiddens / num_heads)
    # we want output.shape=(batch_size, num_queries, num_hiddens)
    x = x.reshape(-1, x.shape[1], x.shape[2] * num_heads)

    return x


def check_MultiHeadAttention():
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.w_k = nn.Linear(key_size, num_hiddens)
        self.w_q = nn.Linear(query_size, num_hiddens)
        self.w_v = nn.Linear(value_size, num_hiddens)
        self.w_o = nn.Linear(num_hiddens, num_hiddens)
        self.attention_score_function = DotProductAttention(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        # queriese.shape=(batch_size, num_queries, q_size)
        # keys.shape=(batch_size, num_keys, k_size)
        # values.shape=(batch_size, num_values, v_size)
        # valid_lens.shape=(batch_size, ) or (batch_size, num_queries)
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        # queriese.shape=(batch_size * num_heads, num_queries, num_hiddens / num_heads)
        # keys.shape=(batch_size * num_heads, num_keys, num_hiddens / num_heads)
        # values.shape=(batch_size * num_heads, num_values, num_hiddens / num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
            # valid_lens.shape=(batch_size * num_heads, ) or valid.shape=(batch_size * num_heads, num_queries)

        output = self.attention_score_function(queries, keys, values, valid_lens)
        self.attention_weights = self.attention_score_function.attention_weights
        output = transpose_output(output, self.num_heads) # output.shape=(batch_size, num_queries, num_hiddens)
        res = self.w_o(output) # res.shape=(batch_size, num_queries, v_size)

        return res


class Seq2SeqMAtionDecoder(Seq2SeqAttentionDecoder):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers, num_heads, dropout=0):
        super().__init__(vocab_size, embedding_dim, num_hiddens, num_layers, dropout=dropout)
        self.attention = MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens, 
                                            num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout)


def main(batch_size=64, epochs=300, lr=0.005, 
         num_steps=10, embedding_dim=32, num_hiddens=32, num_layers=2, dropout=0.1,
         fig_name='multi_head_attention', to_predict=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, source_vocab, target_vocab = load_data_iter(num_steps, batch_size)
    encoder = Seq2SeqEncodcer(vocab_size=len(source_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    decoder = Seq2SeqMAtionDecoder(vocab_size=len(target_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, num_heads=4, dropout=dropout)
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