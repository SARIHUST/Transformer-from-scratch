# --------------------------------------------------------
# Written by Hanhui Wang
# Email: hanhuiwang1108@gmail.com
# This file implements the encoder layer and decoder layer
# used in Transformer's encoder and decoders.
# --------------------------------------------------------

import torch
from torch import nn
from modules import MultiheadAttention

class EncoderLayer(nn.Module):
    '''
    EncoderLayer consists of a multihead self-attention block and a feed forward block.
    '''
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.1, activation='relu', norm_first=False) -> None:
        '''
        Args:
            d_model: encoder dimension
            num_heads: number of heads in multihead attention
            dim_feedforward: dimension in the feed forward module
            dropout: dropout ratio
            activationn: activation function, should be relu or gelu
            norm_first: perform layer normalization before or after attention and feedforward
        '''
        super().__init__()
        assert activation == 'relu' or activation == 'gelu', f'activation function {activation} should be relu or gelu'
        self.norm_first = norm_first

        # self-attention module
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        
        # feed forward module
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # norm blocks
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout after attention and feed forward
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(self, x, attn_mask=None):
        '''
        Wrap up self-attention and dropout for better normalization format
        '''
        x = self.self_attn(x, x, x, attn_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        '''
        Wrap up feed forward and dropout for better normalization format
        '''
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x, attn_mask=None):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))
        return x
            
class DecoderLayer(nn.Module):
    '''
    DecoderLayer consists of a multihead self-attention block, a multihead cross-attention block,
    and a feed forward block.
    '''
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.1, activation='relu', norm_first=False) -> None:
        '''
        Args:
            d_model: encoder dimension
            num_heads: number of heads in multihead attention
            dim_feedforward: dimension in the feed forward module
            dropout: dropout ratio
            activationn: activation function, should be relu or gelu
            norm_first: perform layer normalization before or after attention and feedforward
        '''
        super().__init__()
        assert activation == 'relu' or activation == 'gelu', f'activation function {activation} should be relu or gelu'
        self.norm_first = norm_first

        # self-attention module
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        
        # cross-attention module
        self.cross_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # feed forward module
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # norm blocks
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout after attention and feed forward
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _sa_block(self, x, attn_mask=None):
        '''
        Wrap up self-attention and dropout for better normalization format
        '''
        x = self.self_attn(x, x, x, attn_mask)
        return self.dropout1(x)

    def _ca_block(self, tgt, memory, memory_mask=None):
        x = self.cross_attn(tgt, memory, memory, memory_mask)
        return self.dropout2(x)

    def _ff_block(self, x):
        '''
        Wrap up feed forward and dropout for better normalization format
        '''
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        '''
        Args:
            tgt: the target embedding sequence, tensor shape (b, n_tgt, d) where b is the batch
            size, n_tgt is the sequence lenght, and d is the dimension (equal to d_model)
            memory: the source encoding results (usually drawn from an encoder), tensor shape
            (b, n_memo, d), where b is the batch size, n_memo is the sequence length, and d is
            the dimension (equal to d_model)
            tgt_mask: mask shape (b, n_tgt, n_tgt) or (n_tgt, n_tgt)
            src_mask: mask shape (b, 1, n_src) or (1, n_src)
        '''
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._ca_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._ca_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
        return x
    
if __name__ == '__main__':
    enc_layer = EncoderLayer(256, 8)
    src = torch.randn(4, 10, 256)
    src_mask = torch.randn(4, 1, 10)
    src_mask = (src_mask > 0.2)
    enc_output = enc_layer(src, src_mask)
    print(enc_output.shape)
    dec_layer = DecoderLayer(256, 8)
    tgt = torch.randn(4, 15, 256)
    dec_output = dec_layer(tgt, enc_output)
    print(dec_output.shape)