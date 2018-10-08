import torch


class Monitor:

    def __init__(self):
        self.total = 0
        self.num_batches = 0

    @property
    def value(self) -> float:
        return self.total / self.num_batches if self.num_batches else float('nan')

    def update(self,
               batch_mean: torch.Tensor) -> None:
        self.total += batch_mean.item()
        self.num_batches += 1

    def reset(self):
        self.total = 0
        self.num_batches = 0
