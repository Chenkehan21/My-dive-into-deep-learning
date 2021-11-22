import sys
sys.path.append('../')

import torch
from torch import optim
import torch.nn as nn

from Recurrent_Neural_Network.read_long_sequence_data import load_data_time_machine
from utils.utils import RNN_FROM_SCRATCH, train_rnn


def init_params(vocab_size, hidden_size, device):
    
    def init_block():
        w1 = torch.normal(mean=0, std=0.01, size=(vocab_size, hidden_size), requires_grad=True, device=device)
        w2 = torch.normal(mean=0, std=0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
        b = torch.zeros(size=(hidden_size,), requires_grad=True, device=device)

        return w1, w2, b
    
    w_xi, w_hi, b_i = init_block() # input gate
    w_xf, w_hf, b_f = init_block() # forget gate
    w_xo, w_ho, b_o = init_block() # output gate
    w_xc, w_hc, b_c = init_block() # candidate memory
    
    # final output
    w_hq = torch.normal(mean=0, std=0.01, size=(hidden_size, vocab_size), requires_grad=True, device=device)
    b_q = torch.zeros(size=(vocab_size,), requires_grad=True, device=device)
    
    params = [w_xi, w_hi, b_i, 
              w_xf, w_hf, b_f, 
              w_xo, w_ho, b_o, 
              w_xc, w_hc, b_c, 
              w_hq, b_q]

    return params


def init_state(batch_size, hidden_size, device):
    hidden_state = torch.zeros((batch_size, hidden_size), device=device)
    memory_state = torch.zeros((batch_size, hidden_size), device=device)
    
    return hidden_state, memory_state


def lstm(x_batch, state, params, device):
    w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hq, b_q = params
    H, C = state
    outputs = []
    for x in x_batch:
        x = x.to(device)
        F = torch.sigmoid(x @ w_xf + H @ w_hf + b_f)
        I = torch.sigmoid(x @ w_xi + H @ w_hi + b_i)
        C_tilda = torch.tanh(x @ w_xc + H @ w_hc + b_c)
        O = torch.sigmoid(x @ w_xo + H @ w_ho + b_o)
        
        H = O * torch.tanh(C * F + C_tilda * I)
        C = C * F + C_tilda * I
        
        Y = H @ w_hq + b_q
        
        outputs.append(Y)
    
    res = torch.cat(outputs, dim=0)
    
    return res, (H, C)
    

def main(batch_size=32, epochs=500, lr=1.0, num_steps=32, hidden_size=256):
    data_iter, vocab = load_data_time_machine(batch_size, num_steps, 
                                              use_random_sample=False, 
                                              token='char', max_tokens=-1)
    vocab_size = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = init_params(vocab_size, hidden_size, device)
    
    net = RNN_FROM_SCRATCH(vocab_size, hidden_size, params, init_state, lstm, device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.params, lr)
    trainer = train_rnn()
    
    trainer(epochs, data_iter, vocab, 
            net, loss_fun, optimizer, device, 
            need_to_clip=True, fig_name='lstm_from_scratch', need_to_predict=True)


if __name__ == "__main__":
    main(epochs=2)