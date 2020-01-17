from pathlib import Path
from typing import Any, Union, Optional, Callable, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IngestDataset(Dataset):
    def __init__(
            self, root_dir: str, split: str, transform: Optional[Callable[[np.array], Any]] = None
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.transform = transform

        csv_paths = [p for p in self.root_dir.iterdir() if p.suffix == '.csv']
        if len(csv_paths) > 1:
            raise IOError('Several csv files in the given folder')
        self.dataframe = pd.read_csv(csv_paths[0], index_col=0)
        self.dataframe = self.dataframe.filter(regex=f'^{split}', axis=0)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.array, float, str]]:
        # TODO: update return types
        filepath = self.root_dir / self.dataframe.index[idx]
        # TODO: update
        sample = {
            'features': np.load(filepath),
            'target': self.dataframe.iloc[idx, 0],
            'filename': filepath.name.rsplit('.', 1)[0],
        }

        if self.transform:
            sample['features'] = self.transform(sample['features'])

        return sample


class NpyDataset(Dataset):
    ACCEPTED_EXTENSIONS = ('.npy',)  # TODO: update

    def __init__(
            self, root_dir: str, split: str, transform: Optional[Callable[[Dict], Dict]] = None
    ):
        self._features_shape = None

        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.transform = transform

        split_path = self.root_dir / self.split
        self.filepaths = tuple(
            sorted(e for e in split_path.iterdir() if e.suffix in NpyDataset.ACCEPTED_EXTENSIONS)
        )

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.array, torch.Tensor]]:
        # TODO: update return types
        filepath = self.filepaths[idx]
        # TODO: update
        features, target = np.load(filepath, allow_pickle=True)
        sample = {'features': features.astype(np.float32), 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def features_shape(self):
        if self._features_shape is None:
            self._features_shape = self[0]['features'].shape
        return self._features_shape
