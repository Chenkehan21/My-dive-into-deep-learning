import sys
sys.path.append('../')

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attention_scoring_functions import AdditiveAttention
from Modern_RNNs.seq2seq import *


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # q, k, v are all rnn output's last state
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(num_hiddens + embedding_dim, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, encoder_output_state, encoder_valid_lens):
        outputs, hidden_state = encoder_output_state
        # outputs.shape=(num_steps, batch_size, hidden_size)
        # hidden_state.shape=(D*num_layers, N, H) D=2 if bidirectional=True otherwise 1
        
        return outputs.permute(1, 0, 2), hidden_state, encoder_valid_lens

    def forward(self, x, state):
        encoder_outputs, hidden_state, encoder_valid_lens = state
        X = self.embedding(x).permute(1, 0, 2) # X.shape=(num_steps, batch_sizes, embedding_dim)
        outputs, self._attention_weights = [], []
        for x in X:
            '''
            shape requirements:
            keys.shape=(batch_size, num_keys, k_size)
            queriese.shape=(batch_size, num_queries, q_size)
            values.shape=(batch_size, num_values, v_size)
            ''' 

            queries = hidden_state[-1]
            # now queries.shape=(batch_size, num_hiddens)
            # since we are traversing X on num_steps dimension, x.shape=(batch_sizes, embedding_dim)
            # we can consider num_queries as 1, so we can unsqueeze x on dimension 1
            queries = queries.unsqueeze(dim=1)
            keys = encoder_outputs # keys.shape=(batch_size, num_steps, num_hiddens)
            values = encoder_outputs
            context = self.attention(queries, keys, values, encoder_valid_lens)
            x = torch.cat([context, x.unsqueeze(dim=1)], dim=-1)
            output, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) # output.shape=(num_steps, batch_size, hidden_size)
            outputs.append(output)
            self._attention_weights.append(self.attention.attention_weights)
        
        res = self.dense(torch.cat(outputs, dim=0)).permute(1, 0, 2)
        
        return res, [encoder_outputs, hidden_state, encoder_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights


def main(batch_size=64, epochs=300, lr=0.005, 
         num_steps=10, embedding_dim=32, num_hiddens=32, num_layers=2, dropout=0.1,
         fig_name='bahdanau_attention', to_predict=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, source_vocab, target_vocab = load_data_iter(num_steps, batch_size)
    encoder = Seq2SeqEncodcer(vocab_size=len(source_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    decoder = Seq2SeqAttentionDecoder(vocab_size=len(target_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
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
        plt.savefig('./images/bahdanau_attention_weight.png')
        plt.close()

if __name__ == "__main__":
    main()