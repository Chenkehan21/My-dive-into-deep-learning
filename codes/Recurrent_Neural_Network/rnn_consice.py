import torch
import torch.nn as nn
import torch.nn.functional as F
from read_long_sequence_data import load_data_time_machine
from rnn_from_scratch import predict, train, grad_clip


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.RNN(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, x, hidden_state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32).to(self.device)
        y, hidden_state = self.rnn(x, hidden_state) # y.shape=(num_steps, batch_size, hidden_size)
        y = self.linear(y.reshape(-1, y.shape[-1]))
        
        return y, hidden_state

    def hidden_state_init(self, batch_size):
        return torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size).to(self.device)
    

def main(batch_size=32, lr=0.1, epochs=800, 
        step=35, use_random_sample=False, token='char', 
        need_to_clip=False, fig_name=None, need_to_predict=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data_iter, vocab = load_data_time_machine(batch_size, step, use_random_sample, token)
    net = RNN(len(vocab), 512, device).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    train(epochs, data_iter, vocab, net, loss_fun, optimizer, 
          device, need_to_clip, fig_name, need_to_predict)


if __name__ == "__main__":
    main(epochs=800, fig_name='rnn_consice', need_to_predict=True, need_to_clip=True)