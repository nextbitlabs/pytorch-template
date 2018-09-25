import os
from typing import Tuple

import numpy as np
import torch


class ToFile:

    def __init__(self,
                 output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def __call__(self,
                 sample: Tuple[np.array, float, str]) -> Tuple[np.array, float, str]:
        output_path = os.path.join(self.output_dir, '{}.npy'.format(sample[2]))
        np.save(output_path, np.array(sample[:2]))
        return sample


class ToTensor:

    def __call__(self,
                 sample: Tuple[np.array, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        # noinspection PyCallingNonCallable
        return torch.from_numpy(sample[0]), torch.tensor(sample[1])
