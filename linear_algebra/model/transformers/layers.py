import torch
from torch import nn, Tensor


class AddNorm(nn.Module):
    def __init__(self, shape: int, dropout=.5) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, Y: Tensor):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, num_hiddens: int, num_outputs: int) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(num_outputs)

    def forward(self, X: Tensor) -> Tensor:
        X = self.dense1(X)
        X = self.relu(X)
        X = self.dense2(X)

        return X


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.P = nn.Parameter(torch.zeros(
            (1, max_len, num_hiddens)), requires_grad=False)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: Tensor) -> Tensor:
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)
