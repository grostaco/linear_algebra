from torch import nn, Tensor
from .protocols import Encoder, Decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X: Tensor, dec_X: Tensor, valid_lens: Tensor):
        context = self.encoder(enc_X, valid_lens)
        dec_state = self.decoder.init_state(context, valid_lens)

        dec_outputs, *_ = self.decoder(dec_X, dec_state)
        return dec_outputs
