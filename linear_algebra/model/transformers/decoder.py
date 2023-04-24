import torch
import math

from torch import nn, Tensor
from .layers import AddNorm, PositionWiseFFN, PositionalEncoding
from .utils import as_mask


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_num_hiddens: int, num_heads: int, dropout: float, i: int, bias=False) -> None:
        super().__init__()
        self.i = i

        self.attention1 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, batch_first=True)

        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.attention2 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, batch_first=True)
        self.addnorm2 = AddNorm(embed_dim, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, embed_dim)
        self.addnorm3 = AddNorm(embed_dim, dropout)

    def forward(self, X: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        if self.training:
            _, num_steps, _ = X.shape

            dec_valid_lens = as_mask(torch.arange(
                1, num_steps + 1, device=X.device), key_values.size(1))

        else:
            dec_valid_lens = None

        X2, self._attention_weights = self.attention1(
            X, key_values, key_values, attn_mask=dec_valid_lens, need_weights=True)

        Y = self.addnorm1(X, X2)

        Y2, _ = self.attention2(
            Y, enc_outputs, enc_outputs, key_padding_mask=enc_valid_lens)

        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, num_hiddens: int, ffn_num_hiddens: int, num_heads: int,
                 num_blks: int, dropout: float):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
