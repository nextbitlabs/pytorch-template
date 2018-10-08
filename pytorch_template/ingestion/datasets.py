import os
from typing import Union, Optional, Callable, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IngestDataset(Dataset):

    def __init__(self,
                 root_dir: str,
                 split: str,
                 targets_file: str,
                 transform: Optional[Callable[[Dict], Dict]] = None):
        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.transform = transform

        self.dataframe = pd.read_csv(os.path.join(self.root_dir, targets_file), index_col=0)
        self.dataframe = self.dataframe.filter(regex='^{}'.format(split), axis=0)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self,
                    idx: int) -> Dict[str, Union[np.array, float, str]]:
        # TODO: update return types
        filepath = os.path.join(self.root_dir, self.dataframe.index[idx])
        # TODO: update
        sample = {
            'features': np.load(filepath),
            'target': self.dataframe.iloc[idx, 0],
            'filename': os.path.basename(filepath).rsplit('.', 1)[0]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class NpyDataset(Dataset):
    ACCEPTED_EXTENSIONS = ('npy',)  # TODO: update

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform: Optional[Callable[[Dict], Dict]] = None):
        self._features_shape = None

        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.transform = transform

        split_path = os.path.join(self.root_dir, self.split)
        self.filepaths = tuple(sorted(
            os.path.join(split_path, f) for f in os.listdir(split_path) if
            f.rsplit('.', 1)[1].lower() in NpyDataset.ACCEPTED_EXTENSIONS))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self,
                    idx: int) -> Dict[str, Union[np.array, torch.Tensor]]:
        # TODO: update return types
        filepath = self.filepaths[idx]
        # TODO: update
        features, target = np.load(filepath)
        sample = {
            'features': features.astype(np.float32),
            'target': target
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def features_shape(self):
        if self._features_shape is None:
            self._features_shape = self[0]['features'].shape
        return self._features_shape
