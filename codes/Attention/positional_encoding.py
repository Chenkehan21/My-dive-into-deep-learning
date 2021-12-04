import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from d2l import torch as d2l


class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = torch.zeros(1, max_len, encoding_dim)
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(1000, torch.arange(0, encoding_dim, 2, dtype=torch.float32) / encoding_dim)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)
        
    def forward(self, x):
        x += self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)
    

def check_PositionalEncoding():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.p[:, :X.shape[1], :]
    plt.plot(torch.arange(num_steps), P[0, :, 6].T, label='col6')
    plt.plot(torch.arange(num_steps), P[0, :, 7].T, label='col7')
    plt.plot(torch.arange(num_steps), P[0, :, 8].T, label='col8')
    plt.plot(torch.arange(num_steps), P[0, :, 9].T, label='col9')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/positional_encoding.png')

if __name__ == "__main__":
    check_PositionalEncoding()