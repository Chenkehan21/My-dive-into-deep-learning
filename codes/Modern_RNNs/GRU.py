import sys
sys.path.append('../')

import torch
import torch.nn as nn
from utils.utils import RNN_FROM_SCRATCH, train_rnn


def init_block(vocab_size, hidden_size, device):
    w1 = torch.normal(mean=0, std=0.01, size=(vocab_size, hidden_size), requires_grad=True, device=device)
    w2 = torch.normal(mean=0, std=0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
    b = torch.zeros(size=(hidden_size,), requires_grad=True)

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
    b_q = torch.zeros(size=(hidden_size,), requires_grad=True)

    params = [w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hq, b_q]

    return params


def init_state(batch_size, hidden_size, device):
    return torch.zeros((batch_size, hidden_size), device=device)


def gru(x_batch, hidden_state, params):
    w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hq, b_q = params
    outputs = []
    H = hidden_state
    for x in x_batch:
        R = F.sigmoid(torch.matmul(x, w_xr) + torch.matmul(H, w_hr) + b_r)
        Z = F.sigmoid(torch.matmul(x, w_xz) + torch.matmul(H, w_hz) + b_z)
        H_tilda = F.tanh(torch.matmul(x, w_xh) + torch.matmul(H@R, w_hh) + b_h)
        H = H @ Z + (1 - Z) @ H_tilda
        Y = torch.matmul(H, w_hq) + b_q
        outputs.append(Y)

    res = torch.cat(outputs, dim=1)

    return res, H

    
def main():
    ...