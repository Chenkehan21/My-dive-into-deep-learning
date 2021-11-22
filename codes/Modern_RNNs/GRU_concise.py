import sys
sys.path.append('../')

import torch
from torch import optim
import torch.nn as nn

from Recurrent_Neural_Network.read_long_sequence_data import load_data_time_machine
from utils.utils import RNN_Moduel, train_rnn


def main(batch_size=32, epochs=500, lr=1.0, num_steps=32, hidden_size=256):
    data_iter, vocab = load_data_time_machine(batch_size, num_steps, 
                                              use_random_sample=False, 
                                              token='char', max_tokens=-1)
    vocab_size = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru = nn.GRU(vocab_size, hidden_size)
    net = RNN_Moduel(gru, vocab_size, hidden_size, device).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr)
    trainer = train_rnn()
    
    trainer(epochs, data_iter, vocab, 
            net, loss_fun, optimizer, device, 
            need_to_clip=True, fig_name='gru_concise', need_to_predict=True)
    

if __name__ == '__main__':
    main(epochs=500)