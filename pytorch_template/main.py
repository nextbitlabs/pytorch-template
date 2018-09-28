import logging
import os
import pickle
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .ingestion.datasets import IngestDataset, NpyDataset
from .ingestion.transforms import Normalize, ToFile, ToTensor
from .models.linear import LinearRegression
from .models.model import Model


# TODO: update class name
class PyTorchTemplate:

    @staticmethod
    def ingest(root_dir: str,
               split: str,
               workers: int) -> str:
        # TODO: update transformations
        normalize = Normalize(0, 1)
        to_file = ToFile(os.path.join(root_dir, 'npy', split))
        transformation = transforms.Compose([normalize, to_file])

        dataset = IngestDataset(root_dir, split, 'targets.csv',
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

    @staticmethod
    def train(npy_dir: str,
              output_dir: str,
              batch_size: int,
              epochs: int,
              lr: float,
              workers: int) -> str:
        working_env = PyTorchTemplate._create_working_env(output_dir)

        logging.info('Batch size: {}'.format(batch_size))
        logging.info('Learning rate: {}'.format(lr))
        logging.info('Workers: {}'.format(workers))

        to_tensor = ToTensor()

        train_dataset = NpyDataset(npy_dir, 'train', transform=to_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers)

        dev_dataset = NpyDataset(npy_dir, 'dev', transform=to_tensor)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers)

        module = LinearRegression(train_dataset.features_shape[-1])
        model = Model(module)
        best_checkpoint_path = model.fit(
            working_env, train_loader, epochs, lr, dev_loader)
        return best_checkpoint_path

    @staticmethod
    def evaluate(checkpoint: str,
                 npy_dir: str,
                 batch_size: int,
                 workers: int) -> Tuple[float, float]:
        dev_dataset = NpyDataset(npy_dir, 'dev', transform=ToTensor())
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers)

        module = LinearRegression(dev_dataset.features_shape[-1])
        module.load_state_dict(torch.load(checkpoint))

        model = Model(module)
        val_loss, val_metric = model.eval(dev_loader)
        return val_loss, val_metric

    @staticmethod
    def _create_working_env(output_dir: str) -> str:
        working_env = os.path.join(output_dir, 'runs', str(int(time.time())))
        os.makedirs(os.path.join(working_env, 'checkpoints'))
        os.makedirs(os.path.join(working_env, 'logs'))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler = logging.FileHandler(os.path.join(working_env, 'logs',
                                                        'train_info.log'))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

        return working_env
