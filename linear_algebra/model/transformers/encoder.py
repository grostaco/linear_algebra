
from torch import nn, Tensor
from .layers import AddNorm, PositionWiseFFN, PositionalEncoding

import math
from collections import OrderedDict
from typing import overload, Literal, cast


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_num_hiddens: int, num_heads: int, dropout: float, bias=False) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, batch_first=True)

        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, embed_dim)
        self.addnorm2 = AddNorm(embed_dim, dropout)

    @overload
    def forward(self, X: Tensor, valid_lens: Tensor,
                need_weights: Literal[False]) -> Tensor: ...

    @overload
    def forward(self, X: Tensor, valid_lens: Tensor,
                need_weights: Literal[True]) -> tuple[Tensor, Tensor]: ...

    def forward(self, X: Tensor, valid_lens: Tensor, need_weights=False):
        attn_output, attn_weights = self.attention(
            X, X, X, key_padding_mask=valid_lens, need_weights=need_weights)
        X = self.addnorm1(X, attn_output)
        X = self.addnorm2(X, self.ffn(X))

        if need_weights:
            return X, attn_weights
        return X


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_hiddens: int, ffn_num_hiddens: int, num_heads: int,
                 num_blks: int, dropout: float, bias=False) -> None:
        super().__init__()

        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential(
            OrderedDict((f'block_{i}', TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, bias)) for i in range(num_blks))
        )
        self.attention_weights: list[Tensor | None] = [None] * len(self.blks)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        for i, blk in enumerate(self.blks):
            blk = cast(TransformerEncoderBlock, blk)

            X, weights = blk.forward(X, valid_lens, need_weights=True)
            self.attention_weights[i] = weights

        return X
