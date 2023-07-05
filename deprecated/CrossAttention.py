from torch import nn
from transformer.modules import ScaledDotProductAttention

class CrossAttention(nn.Module):
    '''
    Multi-head self-attention layer, deprecated
    '''
    def __init__(self, embed_dim, head=8, has_mask=False, bias=False) -> None:
        super().__init__()
        assert embed_dim % head == 0, f'Embedding size {embed_dim} should be divisible by number of heads {head}'

        self.embed_dim =  embed_dim
        self.head = head
        self.has_mask = has_mask
        self.head_dim = self.embed_dim // self.head

        # q, k, v are dealed with different projections and then split to different heads
        # this multi-head attention implementation might be slightly different from the original paper
        self.to_queries = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.to_keys = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.to_values = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.attention = ScaledDotProductAttention(self.embed_dim ** 0.5)

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, mask=None):
        '''
        Args:
            query: the query sequence tensor, shape (b, n, d)
            key: the key sequence tensor, shape (b, n, d)
            value: the value sequence tensor, shape (b, n, d)
            mask: tensor shape(b, h, d, d)
        '''
        b, lq, d = query.shape
        assert d == self.embed_dim, f'input dimension {d} should match layer embedding dimension {self.embed_dim}'
        _, l, _ = value.shape   # lk = lv = l

        # project x to q, k, v
        # change the shape to fit multi-head attention
        # (b, lq, d) -> (b, lq, h, d')
        # (b, l, d) -> (b, l, h, d')
        q = self.to_queries(query).reshape(b, lq, self.head, self.head_dim)
        k = self.to_keys(key).reshape(b, l, self.head, self.head_dim)
        v = self.to_values(value).reshape(b, l, self.head, self.head_dim)

        if self.has_mask:
            assert mask != None, 'this self attention layer requires mask'
        else:
            assert mask == None, 'this self attention layer does not require mask'

        attn_output, _ = self.attention(q, k, v, mask=mask)
        attn_output = attn_output.reshape(b, lq, d)  # reshape the tensor back to (b, lq, d)

        output = self.fc(attn_output)
        return output