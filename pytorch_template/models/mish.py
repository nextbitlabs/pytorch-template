# from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt
# implements the mish activation function described in: https://arxiv.org/abs/1908.08681

import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)
