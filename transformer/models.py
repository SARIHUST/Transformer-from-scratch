# ----------------------------------------------------------
# Written by Hanhui Wang
# Email: hanhuiwang1108@gmail.com
# This file implements the Encoder, Decoder, and Transformer
# class in the paper Attention is all you need.
# ----------------------------------------------------------

import torch
from torch import nn
from layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    '''
    Mask out all the padding items in the input sequence tensor.
    Args:
        seq: input sequence tensor, shape (B, L)
        pad_idx: padding idx item
    Return:
        pad_mask: mask tensor, shape (B, 1, L)
    '''
    pad_mask = (seq != pad_idx).unsqueeze(1)
    return pad_mask

def get_subsequent_mask(seq):
    '''
    Generate the triangular mask to mask out subsequent input.
    Args:
        seq: input sequence tensor, shape (B, L)
    Return:
        subsequent_mask: mask tensor, shape (1, L, L)
    '''
    _, L = seq.shape
    subsequent_mask = torch.tril(torch.ones(L, L)).unsqueeze(0).bool()
    return subsequent_mask

class Encoder(nn.Module):
    '''
    Encoder is a stack of N encoder layers.
    '''
    def __init__(self, n_src_vocab, seq_len, d_model, num_layers, num_heads, hidden_dim=1024, pad_idx=0, dropout=0.1, norm=None) -> None:
        '''
        Args:
            n_src_vocan: the total number of possible vocabulary in source sequences
            seq_len: the maximum source sequence lence
            d_model: the embedding dimension
            num_layers: number of encoder layers used
            num_heads: the number of heads in multi-head attention
            hidden_dim: the dimension in the feed forward module
            pad_idx: the padding item where the sequence length is smaller than seq_len
            dropout: dropout ratio
            norm: the layer normalization component (optional)
        '''
        super().__init__()

        self.seq_embedding = nn.Embedding(n_src_vocab, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(self, x, mask):
        # prepare the input to encoder layers
        n, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(n, seq_length)
        enc_input = self.pos_embedding(position) + self.seq_embedding(x)
        enc_output = self.dropout(enc_input)

        # pass through encoder layers
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, mask)

        if self.norm is not None:
            enc_output = self.norm(enc_output)
        return enc_output

class Decoder(nn.Module):
    '''
    Decoder is a stack of N decoder layers.
    '''
    def __init__(self, n_tgt_vocab, seq_len, d_model, num_layers, num_heads, hidden_dim=1024, pad_idx=0, dropout=0.1, norm=None) -> None:
        '''
        Args:
            n_tgt_vocab: the total number of possible vocabulary in target sequences
            seq_len: the maximum target sequence lence
            d_model: the embedding dimension
            num_layers: number of encoder layers used
            num_heads: the number of heads in multi-head attention
            hidden_dim: the dimension in the feed forward module
            pad_idx: the padding item where the sequence length is smaller than seq_len
            dropout: dropout ratio
            norm: the layer normalization component (optional)
        '''
        super().__init__()
        self.seq_embedding = nn.Embedding(n_tgt_vocab, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # prepare the input to the decoder layers
        n, seq_length = tgt.shape
        position = torch.arange(0, seq_length).expand(n, seq_length)
        dec_input = self.pos_embedding(position) + self.seq_embedding(tgt)
        dec_output = self.dropout(dec_input)

        # pass through decoder layers
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output, memory, tgt_mask, memory_mask)

        if self.norm is not None:
            dec_output = self.norm(dec_output)
        return dec_output        

class Transformer(nn.Module):
    '''
    A standard transformer model based on the paper Attention is all you need, which
    consists of an encoder and a decoder.
    '''
    def __init__(self, n_src_vocab, n_tgt_vocab, src_seq_len, tgt_seq_len, d_model, num_layers, num_heads, hidden_dim=1024, src_pad_idx=0, tgt_pad_idx=0, dropout=0.1, norm=None) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            seq_len=src_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            pad_idx=src_pad_idx,
            dropout=dropout,
            norm=norm
        )

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab,
            seq_len=tgt_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            pad_idx=tgt_pad_idx,
            dropout=dropout,
            norm=norm
        )
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.fc = nn.Linear(d_model, n_tgt_vocab)

    def forward(self, src, tgt):
        src_mask = get_pad_mask(src, self.src_pad_idx)                              # shape (B, 1, L_src)
        tgt_mask = get_pad_mask(tgt, self.tgt_pad_idx) & get_subsequent_mask(tgt)   # shape (B, L_tgt, L_tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        output = self.fc(dec_output)

        return output
    
if __name__ == '__main__':
    model = Transformer(20, 50, 15, 15, 256, 4, 8, 512)
    src = torch.tensor([
        [1, 2, 4, 5, 0, 0, 0],
        [4, 2, 1, 3, 5, 4, 0],
        [5, 2, 4, 1, 3, 0, 0]
    ])
    tgt = torch.tensor([
        [1, 5, 2, 8, 2, 0, 0, 0],
        [3, 4, 2, 0, 0, 0, 0, 0],
        [4, 2, 1, 4, 8, 9, 1, 7]
    ])
    out = model(src, tgt)
    print(out.shape)