import os
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .ingestion.datasets import IngestDataset
from .ingestion.transforms import Normalize, ToFile


class PyTorchTemplate:

    @staticmethod
    def ingest(root_dir: str,
               split: str,
               workers: int) -> str:
        #  TODO: update transformations
        normalize = Normalize(0.5, 0.5)
        to_file = ToFile(os.path.join(root_dir, 'npy', split))
        transformation = transforms.Compose([normalize, to_file])

        dataset = IngestDataset(root_dir, split, 'targets.txt',
                                transform=transformation)
        loader = DataLoader(dataset, num_workers=workers)
        for _ in tqdm(loader, total=len(dataset),
                      desc='Writing {} feature files'.format(split)):
            pass

        #  TODO: remove metadata file if not needed (as here)
        metadata_path = os.path.join(root_dir, 'npy', split, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({'num_files': len(dataset)}, f)
        return metadata_path
