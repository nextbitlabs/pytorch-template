import os
from typing import Union, Dict

import numpy as np
import torch


class Normalize:

    def __init__(self,
                 mean: float,
                 std: float):
        self.mean = mean
        self.std = std

    def __call__(self,
                 sample: Dict[str, Union[np.array, float, str]]
                 ) -> Dict[str, Union[np.array, float, str]]:
        sample['features'] = (sample['features'] - self.mean) / self.std
        return sample


class ToFile:

    def __init__(self,
                 output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def __call__(self,
                 sample: Dict[str, Union[np.array, float, str]]
                 ) -> Dict[str, Union[np.array, float, str]]:
        output_path = os.path.join(
            self.output_dir, '{}.npy'.format(sample['filename']))
        np.save(output_path, np.array([sample['features'], sample['target']]))
        return sample


class ToTensor:

    def __call__(self,
                 sample: Dict[str, Union[np.array, float]]
                 ) -> Dict[str, torch.Tensor]:
        # noinspection PyCallingNonCallable
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        return sample
