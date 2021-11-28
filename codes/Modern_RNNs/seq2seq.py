import sys
sys.path.append('../')

import torch
import torch.nn as nn
from machine_translation import load_data_iter


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
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, num_hiddens, num_layers, dropout=dropout)

    
    def forward(self, x):
        # x.shape=(source_batch_size, source_num_steps)
        embedded_x = self.embedding(x) # shape=(batch_size, num_steps, embedding_dim)
        embedded_x = embedded_x.permute(1, 0, 2)
        
        # if hidden_state=None, nn.GRU will automatically initialize hidden state as zeros
        # hidden_state.shape=(D*num_layers, N, H) D=2 if bidirectional=True otherwise 1; N = batch size
        output, hidden_state = self.rnn(embedded_x)
        
        return output, hidden_state
    

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(num_hiddens + embedding_dim, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, encoder_output_state):
        return encoder_output_state[1]
    
    def forward(self, x, state):
        # x.shape=(target_batch_size, target_num_steps)
        # state.shape=(D*num_layers, N, H) D=2 if bidirectional=True otherwise 1; N = batch size
        # state[-1].shape=(batch_size, hidden_state)
        embedded_x = self.embedding(x) # embedded_x.shape=(target_batch_size, target_num_steps, embedding_dim)
        embedded_x = embedded_x.permute(1, 0, 2)
        
        '''
        To further incorporate the encoded input sequence information, 
        the context variable is concatenated with the decoder input at all the time steps
        '''
        #context.shape=(target_num_steps, batch_size, hidden_size)
        context = state[-1].repeat(embedded_x.shape[0], 1, 1)
        # x_and_context.shape=(target_num_steps, batch_size, hidden_size + embedding_dim)
        x_and_context = torch.cat([embedded_x, context], dim=2)
        # output.shape=(num_steps, batch_size, hidden_size)
        # state.shape=(D*num_layers, N, H) D=2 if bidirectional=True otherwise 1; N = batch size
        output, state = self.rnn(x_and_context, state)
        output = self.dense(output) # output.shape=(num_steps, batch_size, vocab_size)
        output = output.permute(1, 0, 2) # output.shape=(batch_size, num_steps, vocab_size)
        
        return output, state
    

def sequence_mask(x, valid_len, mask_value=0):
    ...
    

class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        
    def forward(self, pred, label, valid_len):
        ...
        

def train():
    ...
    

def predict():
    ...
    

def main():
    ...
        

if __name__ == "__main__":
    # data_iter, source_vocab, target_vocab = load_data_iter()
    # tmp = next(iter(data_iter))
    # print(tmp)
    # x = tmp[0]
    # vocab_size = len(source_vocab)
    # embed_size = 32
    # embedding = nn.Embedding(vocab_size, embed_size)
    # output = embedding(x)
    # print(output, output.shape)
    
    encoder = Seq2SeqEncodcer(10, 8, 16, 2)
    decoder = Seq2SeqDecoder(10, 8, 16, 2)
    encoder.eval()
    decoder.eval()
    x = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(x)
    print(output.shape, state.shape)
    state = decoder.init_state(encoder(x))
    output, state = decoder(x, state)
    print(output.shape, state.shape)