import sys
sys.path.append('../')

import math
import torch
import torch.nn as nn

from Modern_RNNs.seq2seq import sequence_mask


def masked_softmax(scores, valid_len):
    # scores.shape=(batch_size, num_queries, num_keys)
    # valid_len is a one dimension tensor
    if valid_len is None:
        return torch.softmax(scores, dim=-1)
    else:
        shape = scores.shape
        if valid_len.dim() == 1:
            # len(valid_len) should equal to batch_size, each element means the valid len of all queries in this batch
            valid_len = torch.repeat_interleave(valid_len, shape[1])
            # after repeat+interleave, valid_len is still a one dimension tensor, len(valid_len) = batch_size * num_queries
        else:
            # valid_len.shape=(batch_size, num_queries) each element means the valid len of one query
            valid_len = valid_len.reshape(-1) # turn valid_len to a one dimension tensor
        scores = sequence_mask(scores.reshape(-1, shape[-1]), valid_len, mask_value=-1e6)
        res = torch.softmax(scores.reshape(shape), dim=-1) # return to the original shape of scores
        '''
        in sequence_mask, max_len = scores.shape[1] = num_queries
        scores.reshape(-1, shape[-1]).shape=(batch_size * num_queries, num_keys)
        valid_len.shape=(batch_size * num_queries,) so the sequence_mask would work well
        after sequence_mask, scores.shape=(batch_size * num_queries, num_keys)
        res.shape=(batch_size, num_queries, num_keys)
        '''

    return res


def check_masked_softmax():
    res = masked_softmax(torch.rand(4, 3, 5), torch.tensor([1, 2, 3, 4]))
    print(res)
    res2 = masked_softmax(torch.rand(4, 3, 5), torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 1, 2]]))
    print()
    print(res2)


def check_AdditiveAttention():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(k_size=2, q_size=20, num_hiddens=8,
                                dropout=0.1)
    attention.eval()
    res = attention(keys, queries, values, valid_lens) # res.shape=(2, 1, 4) => (batch_size, num_queries, v_size)
    print(res, res.shape)


def check_DotProductAttention():
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    res = attention(queries, keys, values, valid_lens)
    print(res)


class AdditiveAttention(nn.Module):
    def __init__(self, q_size, k_size, num_hiddens, dropout=0.5):
        super().__init__()
        self.w_q = nn.Linear(q_size, num_hiddens)
        self.w_k = nn.Linear(k_size, num_hiddens)
        self.w_v = nn.Linear(num_hiddens, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        # keys.shape=(batch_size, num_keys, k_size)
        # queriese.shape=(batch_size, num_queries, q_size)
        # values.shape=(batch_size, num_values, v_size)
        # num_keys = num_values
        
        features_k, features_q = self.w_k(keys), self.w_q(queries)
        # features_k.shape=(batch_size, num_keys, num_hiddens)
        # features_q.shape=(batch_size, num_queries, num_hiddens)
        
        features_k, features_q = features_k.unsqueeze(1), features_q.unsqueeze(2)
        # features_k.shape=(batch_size,      1     , num_keys, num_hiddens)
        # features_q.shape=(batch_size, num_queries,    1    , num_hiddens)
        
        features = torch.tanh(features_k + features_q)
        # features.shape=(batch_size, num_queries, num_keys, num_hiddens)
        
        scores = self.w_v(features)
        # scores.shape=(batch_size, num_queries, num_keys, 1)
        scores = scores.squeeze(-1)
        # scores.shape=(batch_size, num_queries, num_keys)
        
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights.shape = (batch_size, num_queries, num_keys)
        # values.shape            = (batch_size, num_values, v_size), num_kes = num_values
        
        output = torch.bmm(self.dropout(self.attention_weights), values)
        # output.shape=(batch_size, num_queries, v_size)
        
        return output


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        # keys.shape=(batch_size, num_keys, k_size)
        # queriese.shape=(batch_size, num_queries, q_size)
        # values.shape=(batch_size, num_values, v_size)
        # k_size must be equal to q_size, num_queries = num_keys
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights.shape = (batch_size, num_queries, num_keys)
        res = torch.bmm(self.dropout(self.attention_weights), values)
        # res.shape=(batch_size, num_queries, v_size)

        return res

if __name__ == "__main__":
    check_masked_softmax()
    check_AdditiveAttention()
    check_DotProductAttention()