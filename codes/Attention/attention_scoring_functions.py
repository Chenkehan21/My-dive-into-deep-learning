import sys
sys.path.append('../')

import torch
import torch.nn as nn

from Modern_RNNs.seq2seq import sequence_mask


def masked_softmax(scores, valid_len):
    # scores.shape=(batch_size, num_queries, num_keys)
    # valid_len is a one dimension tensor
    shape = scores.shape 
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len, shape[1])


class AdditiveAttention(nn.Module):
    def __init__(self, q_size, k_size, num_hiddens, dropout=0.5):
        super().__init__()
        self.w_q = nn.Linear(q_size, num_hiddens)
        self.w_k = nn.Linear(k_size, num_hiddens)
        self.w_v = nn.Linear(num_hiddens, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, keys, queries, values, valid_lens):
        # keys.shape=(batch_size, num_keys, k_size)
        # queriese.shape=(batch_size, num_queries, q_size)
        # values.shape=(batch_size, num_values, v_size)
        # num_queries = num_values
        
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
        # values.shape            = (batch_size, num_values, v_size), num_queries = num_values
        
        output = torch.bmm(self.dropout(self.attention_weights), values)
        # output.shape=(batch_size, num_queries, v_size)
        
        return output