import sys
sys.path.append('../')

from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import collections

from .machine_translation import load_data_iter, truncate_pad
from Recurrent_Neural_Network.rnn_from_scratch import grad_clip


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
        encoder_outputs = self.encoder(input, *args)
        state = self.decoder.init_state(encoder_outputs, *args)
        
        return self.decoder.forward(x, state)
    

class Seq2SeqEncodcer(Encoder):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, x, *args):
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
        
    def init_state(self, encoder_output_state, *args):
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
    max_len = x.shape[1]
    '''
    the operation is in essence a broadcast:
    
    if x.shape=(a, b, c):
    x[None, :].shape=(1, a, b, c); 
    x[:, None].shape=(a, 1, b, c); 
    x[:, :, None].shape=(a, b, 1, c)
    ...
    
    actually it's the same as x.unsqueeze(dim)
    x[None, :] <=> x.unsqueeze(dim=0)
    x[:, None] <=> x.unsqueeze(dim=1)
    x[:, :, None] <=> x.unsqueeze(dim=2)
    ...
    
    broadcast:
    1. Each tensor has at least one dimension.
    2. When iterating over the dimension sizes, starting at the trailing dimension, 
    the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    
    shape (1, max_len) + shape(len(valid_len), 1) = shape (len(valid_len), max_len)
    '''
    mask = torch.arange(max_len, device=x.device)[None, :] >= valid_len[:, None]
    x[mask] = mask_value
    
    return x
    

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        '''
        pred is the output of Decoder, pred.shape=(batch_size, num_steps, vocab_size)
        label.shape=(batch_size, num_steps)
        valid_lan.shape=(batch_size, )
        '''
        self.reduction = 'none'
        weights = torch.ones_like(label)
        '''
        weights.shape=(batch_size, num_steps)
        max_len = num_steps
        mask -> (1         , num_steps)
                (batch_size,     1    )
                (batch_size, num_steps)
        weights.shape=(batch_size, num_steps)
        '''
        weights = sequence_mask(weights, valid_len)
        # since we define self.reduction='none', unweighted_loss.shape=(batch_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        
        return weighted_loss
        

def train(epochs, data_iter, target_vocab, 
          net, loss_fun, optimizer, device, 
          fig_name='gru_concise'):
    net.train()
    train_losses = []
    for epoch in range(epochs):
        num_tokens, total_loss = 0, 0.0
        for batch in tqdm(data_iter):
            x, x_len, y, y_len = [x.to(device) for x in batch]
            # teach forcing
            bos = torch.tensor([target_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
            decoder_input = torch.cat([bos, y[:, :-1]], dim=1) # not include '<eos>'
            y_hat, _ = net(x, decoder_input, x_len)
            loss = loss_fun(y_hat, y, y_len).sum()
            total_loss += loss
            loss.backward()
            num_tokens += y_len.sum()
            grad_clip(net, 1)
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_loss / num_tokens
        train_losses.append(train_loss)
        print("epoch: ", epoch + 1, "| train loss: %.3f" % train_loss)
        
    if fig_name:
        steps = list(range(len(train_losses)))
        plt.plot(steps, train_losses, label='train_loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("./images/%s.png" % fig_name)
        
            

def predict(net, device, 
            source_sentence: str, source_vocab, target_vocab, num_steps,
            save_attention_weights=False):
    net.eval()
    res, attention_weight_save = [], []
    
    # preprocess source sentence
    sentence_token = source_vocab[source_sentence.lower().split(' ')] + [source_vocab['<eos>']]
    sentence_valid_len = torch.tensor(len(sentence_token), device=device)
    sentence_token = truncate_pad(sentence_token, num_steps, source_vocab['<pad>'])
    
    encoder_input = torch.unsqueeze(torch.tensor(sentence_token, device=device), dim=0)
    encoder_output = net.encoder(encoder_input)
    decoder_state = net.decoder.init_state(encoder_output, sentence_valid_len)
    decoder_input = torch.unsqueeze(torch.tensor([target_vocab['<bos>']], device=device), dim=0)
    
    for _ in range(num_steps):
        output, decoder_state = net.decoder(decoder_input, decoder_state)
        dec_x = output.argmax(dim=2)
        pred = dec_x.squeeze(dim=0).item()
        if pred == target_vocab['<eos>']:
            break
        if save_attention_weights:
            attention_weight_save.append(net.decoder.attention_weights)
        res.append(pred)
    
    return ''.join(target_vocab.to_tokens(res)), attention_weight_save
    
    
def BLEU(pred_seq, label_seq, k):
    pred_seq, label_seq = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_seq), len(label_seq)
    score = math.exp(min(0, 1 - len_pred / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_seq[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(label_seq[i : i + n])] > 0:
                num_matches += 1
                label_subs[''.join(label_seq[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1 + 1e-5), math.pow(0.5, n))
    
    return score
    

def main(batch_size=64, epochs=300, lr=0.005, 
         num_steps=10, embedding_dim=32, num_hiddens=32, num_layers=2, dropout=0.1,
         fig_name='seq2seq', to_predict=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter, source_vocab, target_vocab = load_data_iter(num_steps, batch_size)
    encoder = Seq2SeqEncodcer(vocab_size=len(source_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    decoder = Seq2SeqDecoder(vocab_size=len(target_vocab), embedding_dim=embedding_dim, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
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
        

if __name__ == "__main__":
    main()