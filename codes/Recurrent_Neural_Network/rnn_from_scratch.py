import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from read_long_sequence_data import load_data_time_machine


class RNN:
    def __init__(self, vocab_size, hidden_size, device):
        input_size = output_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        self.vocab_size = vocab_size

        self.w_xh = torch.normal(mean=0, std=0.01, size=(input_size, hidden_size), requires_grad=True, device=device)
        self.w_hh = torch.normal(mean=0, std=0.01, size=(hidden_size, hidden_size), requires_grad=True, device=device)
        self.b_h = torch.zeros(size=(hidden_size,), requires_grad=True, device=device)

        self.w_hq = torch.normal(mean=0, std=0.01, size=(hidden_size, output_size), requires_grad=True, device=device)
        self.b_q = torch.zeros(size=(output_size,), requires_grad=True, device=device)
        self.params = [self.w_xh, self.w_hh, self.b_h, self.w_hq, self.b_q]

    def __call__(self, x_batch, hidden_state):
        res = []
        H = hidden_state
        X = F.one_hot(x_batch.T, self.vocab_size).type(torch.float32)
        for x in X:
            x = x.to(self.device)
            H = torch.tanh(torch.matmul(x, self.w_xh) + torch.matmul(H, self.w_hh) + self.b_h)
            Y = torch.matmul(H, self.w_hq) + self.b_q
            res.append(Y)
        
        output = torch.cat(res, dim=0)

        return output, H

    def hidden_state_init(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device)


def grad_clip(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad] # net.parameters() returns a generator
    else:
        params = net.params
    norm = torch.sqrt(sum([torch.sum(p.grad**2) for p in params]))
    if norm > theta:
        for param in params:
            param.grad *= theta / norm


def predit(net, text, predict_steps, vocab, device):
    hidden_state = net.hidden_state_init(batch_size=1)
    res = [vocab[text[0]]]
    get_data = lambda: torch.tensor(res[-1], device=device).reshape(1, 1)
    for token in text[1:]:
        _, hidden_state = net(get_data(), hidden_state)
        res.append(vocab[token])
    for i in range(predict_steps):
        output, hidden_state = net(get_data(), hidden_state)
        # print(output.shape)
        # print(output.argmax(dim=1))
        res.append(int(output.argmax(dim=1).reshape(1)))
    
    return ''.join(vocab.to_tokens(i) for i in res)


def train(batch_size=32, lr=0.5, epochs=500, step=35, use_random_sample=False, token='char', plot=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data_iter, vocab = load_data_time_machine(batch_size, step, use_random_sample, token)
    net = RNN(len(vocab), 512, device)
    loss_fun = nn.CrossEntropyLoss()
    # print(net.params[0].is_leaf)
    optimizer = torch.optim.SGD(net.params, lr)

    total_perplexity = []
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for x, y in data_iter:
            hidden_state = net.hidden_state_init(batch_size=x.shape[0])
            output, hidden_state = net(x, hidden_state)
            y = y.T.reshape(-1)
            x, y = x.to(device), y.to(device)
            loss = loss_fun(output, y)
            loss.backward()
            grad_clip(net, 1.0)
            total_loss += loss * y.numel()
            n += y.numel()
            optimizer.step()
            optimizer.zero_grad()
        perplexity = torch.exp(total_loss / n).item()
        total_perplexity.append(perplexity)
        print("epoch: %d|train_loss: %.3f" % (epoch + 1, perplexity))
    
    if plot:
        x = list(range(len(total_perplexity)))
        plt.plot(x, total_perplexity, label='train')
        plt.legend()
        plt.grid(True)
        plt.savefig('./images/rnn_from_scratch_grad_clip.png')

    # predict_text = predit(net, 'time traveller ', 2, vocab, device)
    # print(predict_text)


if __name__ == "__main__":
    train(plot=True)