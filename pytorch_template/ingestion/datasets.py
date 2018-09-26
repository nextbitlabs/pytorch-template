import os
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class IngestDataset(Dataset):

    def __init__(self,
                 root_dir: str,
                 split: str,
                 targets_file: str,
                 transform: Optional[Callable[[np.array], np.array]] = None):
        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.transform = transform

        self.dataframe = pd.read_csv(os.path.join(self.root_dir, targets_file), index_col=0)
        self.dataframe = self.dataframe.filter(regex='^{}'.format(split), axis=0)

    def __len__(self) -> int:
        return len(self.dataframe)

    # TODO: update
    def __getitem__(self,
                    idx: int) -> Tuple[np.array, float, str]:  # TODO: update return types
        filepath = os.path.join(self.root_dir, self.dataframe.index[idx])
        features = np.load(filepath)
        target = self.dataframe.iloc[idx, 0]
        filename = os.path.basename(filepath).rsplit('.', 1)[0]

        sample = (features, target, filename)
        if self.transform:
            sample = self.transform(sample)

        return sample


class NpyDataset(Dataset):
    ACCEPTED_EXTENSIONS = ('npy',)  # TODO: update

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform: Optional[Callable[[np.array], np.array]] = None):
        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.transform = transform

        split_path = os.path.join(self.root_dir, self.split)
        self.filepaths = tuple(sorted(
            os.path.join(split_path, f) for f in os.listdir(split_path) if
            f.rsplit('.', 1)[1].lower() in NpyDataset.ACCEPTED_EXTENSIONS))

    def __len__(self) -> int:
        return len(self.filepaths)

    # TODO: update
    def __getitem__(self,
                    idx: int) -> Tuple[np.array, float]:  # TODO: update return types
        filepath = self.filepaths[idx]
        features, target = np.load(filepath)

        sample = (features, target)
        if self.transform:
            sample = self.transform(sample)

        return sample
