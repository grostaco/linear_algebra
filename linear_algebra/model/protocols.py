from typing import Protocol, TypeVar, overload, Any
from torch import Tensor

T = TypeVar('T')


class Encoder(Protocol):
    @overload
    def __call__(self, src: Tensor) -> Tensor: ...
    @overload
    def __call__(self, src: Tensor, attn_mask: Tensor) -> Tensor: ...


class Decoder(Protocol):
    def init_state(self, context: Tensor, attn_mask: Tensor) -> Any: ...
    def __call__(self, tgt: Tensor, state: Any) -> Tensor: ...
