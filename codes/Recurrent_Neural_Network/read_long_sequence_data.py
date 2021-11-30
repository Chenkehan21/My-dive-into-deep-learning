from .text_preprocess import load_corpus_time_machine
import random
import torch


def seq_data_iter_random(corpus, step, batch_size):
    init_index = random.randint(0, step - 1)
    corpus = corpus[init_index:]
    num_sequence = (len(corpus) - 1) // step
    num_batch = num_sequence // batch_size
    sequence_index = [index for index in range(0, num_sequence * step, step)]
    random.shuffle(sequence_index)
    batch_index = [index for index in range(0, num_batch * batch_size, batch_size)]
    # random.shuffle(batch_index)
    
    generate_data = lambda idx: corpus[idx : idx + step]
    for i in batch_index:
        x = [generate_data(idx) for idx in sequence_index[i : i + batch_size]]
        y = [generate_data(idx + 1) for idx in sequence_index[i : i + batch_size]]
        yield torch.tensor(x), torch.tensor(y)


def seq_data_iter_sequential(corpus, step, batch_size):
    offset = random.randint(0, step)
    num_tokens = (len(corpus) - offset - 1) // batch_size * batch_size
    xs = torch.tensor(corpus[offset : offset + num_tokens])
    ys = torch.tensor(corpus[offset + 1 : offset + num_tokens + 1])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batch = xs.shape[1] // step
    for i in range(0, num_batch * step, step):
        x = xs[:, i : i + step]
        y = ys[:, i : i + step]
        yield x, y


def test():
    test_seq = list(range(35))
    print("random:\n")
    for x, y in seq_data_iter_random(test_seq, step=5, batch_size=2):
        print('x:', x, '\n', 'y:', y, '\n')
    
    print("sequential:\n")
    for x, y in seq_data_iter_sequential(test_seq, step=5, batch_size=2):
        print('x:', x, '\n', 'y:', y, '\n')


class SeqDataLoader:
    def __init__(self, batch_size, step, use_random_sample, token, max_tokens=-1):
        self.step = step
        self.batch_size = batch_size
        if use_random_sample:
            self.sample_fn = seq_data_iter_random
        else:
            self.sample_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(token, max_tokens)
    
    def __iter__(self):
        return self.sample_fn(self.corpus, self.step, self.batch_size)


def load_data_time_machine(batch_size, step, use_random_sample=False, token='char', max_tokens=-1):
    data_iter = SeqDataLoader(batch_size, step, use_random_sample, token)
    return data_iter, data_iter.vocab


if __name__ == "__main__":
    test()