import os
import sys
sys.path.append('../')

import torch
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter

from d2l import torch as d2l
from Recurrent_Neural_Network.text_preprocess import Vocab


def load_raw_data():
    d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()
    
'''
 replace non-breaking space with space, convert uppercase letters to lowercase ones, 
 and insert space between words and punctuation marks.
'''
def preprocess_data(text):
    # 1. Replace non-breaking space with space, and convert uppercase letters to
    # 2. lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    
    # 3. Insert space between words and punctuation marks
    def check_space(char, pre_char):
        return char in set(',.?!') and pre_char != ' '
    
    res = [' ' + char if i > 0 and check_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    
    return ''.join(res)


def tokenize_data(text):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    
    return source, target


def check_distribution():
    text = load_raw_data()
    text = preprocess_data(text)
    print(text[:1000])
    
    source, target = tokenize_data(text)
    print(source[:10])
    print(target[:10])
    
    x = [len(i) for i in source]
    y = [len(i) for i in target]
    count_x = Counter(x)
    count_y = Counter(y)
    print(count_x)
    print(count_y)
    
    _, _, _ = plt.hist([x, y], label=['source', 'target'])
    plt.xlabel('token size')
    plt.ylabel('token number')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/machine_translation_token.png')


def truncate_pad(line, num_steps, padding_token):
    if len(line) < num_steps:
        return line + [padding_token] * (num_steps - len(line))
    return line[:num_steps]
    

def build_array(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    
    return array, valid_len
    

def load_array(array, batch_size, is_train=True):
    dataset = data.TensorDataset(*array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_iter(num_steps=8, batch_size=2, is_train=True):
    text = load_raw_data()
    text = preprocess_data(text)
    source, target = tokenize_data(text)
    
    source_vocab = Vocab(min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'], tokens=source)
    target_vocab = Vocab(min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'], tokens=target)
    
    source_array, source_len = build_array(source, source_vocab, num_steps)
    target_array, target_len = build_array(target, target_vocab, num_steps)
    
    data_arrays = (source_array, source_len, target_array, target_len)
    data_iter = load_array(data_arrays, batch_size, is_train)
    
    return data_iter, source_vocab, target_vocab
    
    
if __name__ == "__main__":
    data_iter, source_vocab, target_vocab = load_data_iter()
    tmp = next(iter(data_iter))
    print(tmp)