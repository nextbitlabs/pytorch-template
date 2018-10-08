from typing import Union, Dict

import numpy as np
import torch


class ToTensor:

    def __call__(self,
                 sample: Dict[str, Union[np.array, float]]
                 ) -> Dict[str, torch.Tensor]:
        # noinspection PyCallingNonCallable
        sample = {k: torch.as_tensor(v) for k, v in sample.items()}
        return sample
