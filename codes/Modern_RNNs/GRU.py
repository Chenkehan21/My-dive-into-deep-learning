import sys
sys.path.append('../')

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from Recurrent_Neural_Network.read_long_sequence_data import load_data_time_machine
from utils.utils import RNN_FROM_SCRATCH, train_rnn


def init_block(vocab_size, hidden_size, device):
    w1 = torch.normal(mean=0, std=0.01, size=(vocab_size, hidden_size), requires_grad=True, device=device)
    w2 = torch.normal(mean=0, std=0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
    b = torch.zeros(size=(hidden_size,), requires_grad=True, device=device)

    return w1, w2, b


def init_params(vocab_size, hidden_size, device):
    # reset gate
    w_xr, w_hr, b_r = init_block(vocab_size, hidden_size, device)

    # update gate
    w_xz, w_hz, b_z = init_block(vocab_size, hidden_size, device)

    # hidden state
    w_xh, w_hh, b_h = init_block(vocab_size, hidden_size, device)

    # output
    w_hq = torch.normal(mean=0, std=0.01, size=(hidden_size, vocab_size), requires_grad=True, device=device)
    b_q = torch.zeros(size=(vocab_size,), requires_grad=True, device=device)

    params = [w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hq, b_q]

    return params


def init_state(batch_size, hidden_size, device):
    return torch.zeros((batch_size, hidden_size), device=device)


def gru(x_batch, hidden_state, params, device):
    w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hq, b_q = params
    outputs = []
    H = hidden_state
    for x in x_batch:
        x = x.to(device)
        R = torch.sigmoid(torch.matmul(x, w_xr) + torch.matmul(H, w_hr) + b_r)
        Z = torch.sigmoid(torch.matmul(x, w_xz) + torch.matmul(H, w_hz) + b_z)
        H_tilda = torch.tanh(torch.matmul(x, w_xh) + torch.matmul(H*R, w_hh) + b_h)
        H = H * Z + (1 - Z) * H_tilda
        Y = torch.matmul(H, w_hq) + b_q
        outputs.append(Y)

    res = torch.cat(outputs, dim=0)

    return res, H

    
def main(batch_size=32, epochs=500, lr=1.0, num_steps=32, hidden_size=256):
    data_iter, vocab = load_data_time_machine(batch_size, num_steps, 
                                              use_random_sample=False, 
                                              token='char', max_tokens=-1)
    vocab_size = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = init_params(vocab_size, hidden_size, device)
    
    net = RNN_FROM_SCRATCH(vocab_size, hidden_size, params, init_state, gru, device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.params, lr)
    trainer = train_rnn()
    
    trainer(epochs, data_iter, vocab, 
            net, loss_fun, optimizer, device, 
            need_to_clip=True, fig_name='gru_from_scratch', need_to_predict=True)


if __name__ == "__main__":
    main(epochs=2)