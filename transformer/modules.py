# -----------------------------------------------------
# Written by Hanhui Wang
# Email: hanhuiwang1108@gmail.com
# This file implements the Scaled dot product attention
# and a certain type of Multi-head attention mechanism.
# -----------------------------------------------------

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    '''
    An implementation of scaled dot product attention described in the paper
    Attention is all you need.
    '''
    def __init__(self, scale, dropout=0) -> None:
        '''
        Args:
            scale: the down scaling factor
            dropout: dropouot ratio
        '''
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        '''
        Args:
            q: tensor shape (b, q, h, d), queries
            k: tensor shape (b, l, h, d), keys
            v: tensor shape (b, l, h, d), values
            key and value should come from the same source, having the same sequence length
            mask: tensor shape (b, h, q, l)
        Return:
            out: the queries after capturing information from k, v
            attn: the computed attention between 
        '''
        # use einsum to deal with complicated batch matrix multiplication
        attn = torch.einsum('bqhd, blhd -> bhql', (q, k))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e20)   # -inf will turn to 0 after softmax

        attn = self.dropout(torch.softmax(attn / self.scale, dim=3))
        out = torch.einsum('bhql, blhd -> bqhd', (attn, v))

        return out, attn
  
class MultiheadAttention(nn.Module):
    '''
    A canonical implementation of multihead attention module according to the paper
    Attention is all you need.
    '''
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0, bias=False) -> None:
        '''
        Args:
            embed_dim: model embedding dimension, should be the same with query embedding dimension
            num_heads: multi-head parameter
            kdim: original key dimension
            vdim: original value dimension
        '''
        super().__init__()
        assert embed_dim % num_heads == 0, f'Embedding size {embed_dim} should be divisible by number of heads {num_heads}'
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.key_projection = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.value_projection = nn.Linear(self.vdim, self.embed_dim, bias=bias)

        self.attention = ScaledDotProductAttention(self.embed_dim ** 0.5, dropout=dropout)
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, mask=None, need_weights=False):
        '''
        Args:
            query: query embeddings of shape (B, L, E_q), where L is the target sequence length
            key: key embeddings of shape (B, S, E_k), where S is the source sequence length
            value: value embeddings of shape (B, S, E_v), where S is the source sequence length
            mask: mask shape of (L, S) or (B, L, S) or (1, S) or (B, 1, S)
            need_weight: if set to True, returns the attention value
        '''
        b, lq, dq = query.shape
        assert dq == self.embed_dim, f'query embedding dimension {dq} should be the same with model embedding dimension {self.embed_dim}'
        _, lk, dk = key.shape
        assert dk == self.kdim, f'key embedding dimension {dk} should be {self.kdim}'
        _, lv, dv = value.shape
        assert dv == self.vdim, f'value embedding dimension {dv} should be {self.vdim}'
        assert lk == lv, f'key sequence length {lk} should be the same with value sequence length {lv}'

        # project query, key, value to q, k, v
        q = self.query_projection(query).reshape(b, lq, self.num_heads, self.head_dim)
        k = self.key_projection(key).reshape(b, lk, self.num_heads, self.head_dim)
        v = self.value_projection(value).reshape(b, lv, self.num_heads, self.head_dim)

        # deal with masks
        if mask is not None:
            if len(mask.shape) == 2:    # mask shape (L, S) or (1, S), add dimension for batch
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1)    # add mask dimension for multiple heads

        q, attn = self.attention(q, k, v, mask)
        q = q.reshape(b, lq, self.embed_dim)

        out = self.linear(q)
        
        if need_weights:
            return out, attn
        else:
            return out
    
if __name__ == '__main__':
    attn = MultiheadAttention(256, 8, kdim=128, vdim=64)
    q = torch.randn(4, 10, 256)
    k = torch.randn(4, 15, 128)
    v = torch.randn(4, 15, 64)
    mask = torch.randn(1, 15)
    mask = (mask > 0.2)
    print(mask)
    output = attn(q, k, v, mask)
    print(output.shape)