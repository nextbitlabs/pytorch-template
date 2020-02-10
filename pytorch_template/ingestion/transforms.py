from typing import Sequence

import torch


class ToTensor:
    def __call__(self, value: float) -> torch.Tensor:
        # noinspection PyCallingNonCallable
        return torch.as_tensor(value)


class Normalize:
    def __init__(
        self, mean: Sequence[float], std: Sequence[float], inplace: bool = False
    ):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.inplace = inplace

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.sub_(self.mean) if self.inplace else t.sub(self.mean)
        t = t.div_(self.std) if self.inplace else t.div(self.std)
        return t
