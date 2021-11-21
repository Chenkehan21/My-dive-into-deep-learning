import collections
import re
from d2l import torch as d2l


d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def load_book():
    with open(d2l.download('time_machine'), 'r') as f:
        text = f.readlines()
        lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in text]
    
    return lines


def tokenize(lines, token='word'):
    if token == 'word':
        tokens = [line.split() for line in lines]
    elif token == 'char':
        tokens = [list(char) for char in lines]
    else:
        print("unknown token: ", token)
        exit()
    
    return tokens


class Vocab:
    def __init__(self, min_freq=0, reserved_tokens=None, tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        freq = calculate_frequence(tokens)
        self.freq_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        self.unk = 0
        res = ['unk'] + reserved_tokens
        res += [token for token, freq in self.freq_tokens if freq > min_freq and token not in res]

        self.index_to_token, self.token_to_index = [], {}
        for i, token in enumerate(res):
            self.index_to_token.append(token)
            self.token_to_index[token] = len(self.index_to_token) - 1

    def __len__(self):
        return len(self.index_to_token)
    
    def __getitem__(self, token):
        if not isinstance(token, (list, tuple)):
            return self.token_to_index.get(token, self.unk)
        return [self.__getitem__(item) for item in token]

    def to_tokens(self, index):
        if not isinstance(index, (list, tuple)):
            return self.index_to_token[index]
        return [self.to_tokens(item) for item in index]


def calculate_frequence(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]

    return collections.Counter(tokens)


def test():
    lines = load_book()
    tokens_word = tokenize(lines, 'word')
    vocab_word = Vocab(tokens=tokens_word)
    print(len(vocab_word))
    print(vocab_word[('the', 'i', 'and', 'like', 'happy', 'tomorrow', 'asldkf', '23', '%^&*')])
    print(vocab_word.to_tokens([1, 2, 3, 100, 1000, 2000, 3000, 4000, 4580-1]))


def load_corpus_time_machine(token='char', max_tokens=-1):
    lines = load_book()
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens=tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab



if __name__ == "__main__":
    test()
    corpus, _ = load_corpus_time_machine()
    print(len(corpus))
    print(corpus[:5])