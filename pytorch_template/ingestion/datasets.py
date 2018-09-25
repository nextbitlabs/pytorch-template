import os
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class IngestDataset(Dataset):
    ACCEPTED_EXTENSIONS = ('npy',)  # TODO: update

    def __init__(self,
                 root_dir: str,
                 split: str,
                 targets_file: str,
                 transform: Optional[Callable[[np.array], np.array]] = None):
        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.targets = pd.read_csv(os.path.join(self.root_dir, targets_file), index_col=0)
        self.transform = transform

        split_path = os.path.join(self.root_dir, self.split)
        self.filepaths = tuple(sorted(
            os.path.join(self.split, f) for f in os.listdir(split_path) if
            f.rsplit('.', 1)[1].lower() in NpyDataset.ACCEPTED_EXTENSIONS))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self,
                    idx: int) -> Tuple[np.array, float, str]:  # TODO: update return types
        filepath = os.path.join(self.root_dir, self.filepaths[idx])
        features = np.load(filepath)
        target = self.targets.loc[self.filepaths[idx]]['target']
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

    def __getitem__(self,
                    idx: int) -> Tuple[str, np.array, int]:  # TODO: update return types
        filepath = self.filepaths[idx]
        features, target = np.load(filepath)

        sample = (features, target)
        if self.transform:
            filename = os.path.basename(filepath).rsplit('.', 1)[0]
            sample = self.transform(sample, filename=filename)

        return sample
