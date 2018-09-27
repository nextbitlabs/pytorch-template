from typing import Optional, Callable

import torch


class Monitor:

    def __init__(self,
                 target_condition: Optional[Callable] = None):
        self.total = 0
        self.num_valid = 0
        self.target_condition = target_condition

    @property
    def value(self) -> float:
        return self.total / self.num_valid if self.num_valid else float('nan')

    def update(self,
               batch_total: torch.Tensor,
               targets: torch.Tensor) -> None:
        with torch.no_grad():
            self.total += batch_total.item()
            batch_valid = sum(self.target_condition(targets)).item() \
                if self.target_condition else targets.shape[0]
            self.num_valid += batch_valid

    def reset(self):
        self.total = 0
        self.num_valid = 0


class AccuracyMonitor(Monitor):

    # noinspection PyTypeChecker
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        with torch.no_grad():
            self.total += sum(predictions == targets).item()
            batch_valid = sum(self.target_condition(targets)).item() \
                if self.target_condition else targets.shape[0]
            self.num_valid += batch_valid
