import matplotlib.pyplot as plt
from text_preprocess import Vocab, load_corpus_time_machine


def show(num, n_grams=[], plot=False):
    corpus, vocab = load_corpus_time_machine('word')
    if isinstance(n_grams, (list, tuple)):
        for n_gram in n_grams:
            start = list(range(n_gram))
            end = [-x for x in reversed(start)][:-1]
            end.append(None)
            slice_group = [pair for pair in zip(start, end)]
            corpus_group = [corpus[a : b] for a, b in slice_group]
            n_gram_tokens = [pair for pair in zip(*corpus_group)]
            vocab = Vocab(tokens=n_gram_tokens)
            freq_tokens = vocab.freq_tokens
            print(freq_tokens[:num], '\n')
            if plot:
                x = list(range(len(freq_tokens)))
                y = [freq for _, freq in freq_tokens]
                plt.loglog(x, y, label='%d-gram'%n_gram)
                plt.legend()
                plt.xlabel('tokens')
                plt.ylabel('frequency')
                plt.grid(True)
                plt.savefig('./images/char_token_frequency_log.png')


if __name__ == "__main__":
    show(10, n_grams=[1, 2, 3, 4, 5], plot=True)