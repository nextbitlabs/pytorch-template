from typing import Optional

import torch


class Monitor:
    def __init__(self, reduction: str = 'none'):
        self.reduction = reduction
        self.cumulated_value = 0
        self.num_samples = 0

    @property
    def value(self) -> float:
        return self.cumulated_value / self.num_samples if self.num_samples else float('nan')

    def update(self, batch_loss: torch.Tensor, batch_size: Optional[int] = None) -> None:
        if batch_size is None and self.reduction != 'none':
            raise ValueError('Missing batch size')
        if self.reduction == 'none':
            if batch_size is not None and batch_loss.size(0) != batch_size:
                raise ValueError('Wrong specified batch size')
            else:
                batch_size = batch_loss.size(0)
            self.cumulated_value += torch.sum(batch_loss).item()
        elif self.reduction == 'mean':
            self.cumulated_value += batch_size * batch_loss.item()
        elif self.reduction == 'sum':
            self.cumulated_value += batch_loss.item()
        else:
            raise ValueError('Unknown reduction method')
        self.num_samples += batch_size

    def reset(self):
        self.cumulated_value = 0
        self.num_samples = 0
