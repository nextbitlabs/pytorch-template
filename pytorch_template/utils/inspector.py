from typing import Sequence, Union

import torch
import torch.nn as nn


def _get_unique_value(value: Union[int, Sequence[int]]) -> int:
    if isinstance(value, Sequence):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                'value must be a pair of integers with the same value or a single integer'
            )
        value = value[0]
    return value


# noinspection PyUnresolvedReferences
def receptive_field(model: nn.Module) -> int:
    layers_sequence = [
        layer
        for layer in model.modules()
        if hasattr(layer, 'stride') and hasattr(layer, 'kernel_size')
    ]
    strides = torch.tensor(
        [_get_unique_value(layer.stride) for layer in layers_sequence]
    )
    kernel_sizes = torch.tensor(
        [_get_unique_value(layer.kernel_size) for layer in layers_sequence]
    )
    cumulated_strides = torch.cat((torch.tensor(1), strides.cumprod(0)[:-1]), 0)
    return int(torch.sum((kernel_sizes - 1) * cumulated_strides) + 1)


def effective_stride(model: nn.Module) -> int:
    # noinspection PyUnresolvedReferences
    strides = torch.tensor(
        [
            _get_unique_value(layer.stride)
            for layer in model.modules()
            if hasattr(layer, 'stride')
        ]
    )
    return int(strides.prod())
