import torch
from torch import Tensor


def as_mask(valid_lens: Tensor, maxlen: int):
    mask = torch.arange(1, maxlen+1, device=valid_lens.device)[:, None]

    return (valid_lens < mask).t()
