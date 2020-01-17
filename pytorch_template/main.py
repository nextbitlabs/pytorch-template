import logging
import multiprocessing
import pickle
import shutil
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .ingestion.datasets import IngestDataset, NpyDataset
from .ingestion.transforms import ToTensor, Normalize
from .models.linear import LinearRegression
from .models.model import Model
from .utils.logger import initialize_logger


# TODO: update class name
class PyTorchTemplate:
    @staticmethod
    def _load_model(checkpoint: str) -> Model:
        with open(Path(checkpoint).parent / 'hyperparams.pkl', 'rb') as f:
            hyperparams = pickle.load(f)

        if LinearRegression.__name__ == hyperparams['module_name']:  # TODO: update module
            module_class = LinearRegression  # TODO: update module
        else:
            raise ValueError('Checkpoint of unsupported module')
        del hyperparams['module_name']

        module = module_class(**hyperparams)
        module.load_state_dict(torch.load(checkpoint))
        model = Model(module)
        return model

    @staticmethod
    def ingest(root_dir: str, split: str) -> str:
        initialize_logger()

        # TODO: update transformations
        transform = Compose(
            [
                ToTensor(),
                # TODO: replace values with statistics computed by dataset_statistics.py
                Normalize(
                    mean=(0.502, 0.475, 0.475, 0.534, 0.493),
                    std=(0.283, 0.277, 0.281, 0.302, 0.306),
                ),
            ]
        )

        dataset = IngestDataset(root_dir, split, transform=transform)
        loader = DataLoader(dataset, batch_size=None, num_workers=multiprocessing.cpu_count())

        # TODO: update path
        output_dir = Path(root_dir) / 'npy' / Path(split)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)

        for sample in tqdm(loader, desc=f'Writing {split} feature files'):
            output_path = output_dir / f"{sample['filename']}.npy"
            np.save(output_path, np.array([sample['features'].numpy(), sample['target']]))

        #  TODO: remove metadata file if not needed (as here)
        metadata_path = Path(root_dir) / 'npy' / split / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump({'num_files': len(dataset)}, f)
        return metadata_path

    @staticmethod
    def train(npy_dir: str, output_dir: str, batch_size: int, epochs: int, lr: float) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        train_dataset = NpyDataset(npy_dir, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        if (Path(npy_dir) / 'dev').is_dir():
            dev_dataset = NpyDataset(npy_dir, 'dev')
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True,
            )
        else:
            dev_loader = None

        module = LinearRegression(train_dataset.features_shape[-1])
        model = Model(module)
        best_checkpoint = model.fit(run_dir, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

    @staticmethod
    def restore(
            npy_dir: str, checkpoint: str, output_dir: str, batch_size: int, epochs: int, lr: float
    ) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Checkpoint: {checkpoint}')
        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        train_dataset = NpyDataset(npy_dir, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        if (Path(npy_dir) / 'dev').is_dir():
            dev_dataset = NpyDataset(npy_dir, 'dev')
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True,
            )
        else:
            dev_loader = None

        model = PyTorchTemplate._load_model(checkpoint)
        best_checkpoint = model.fit(run_dir, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

    @staticmethod
    def evaluate(checkpoint: str, npy_dir: str, batch_size: int) -> Tuple[float, float]:
        dev_dataset = NpyDataset(npy_dir, 'dev')
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        model = PyTorchTemplate._load_model(checkpoint)
        val_loss, val_metric = model.eval(dev_loader)
        return val_loss, val_metric

    @staticmethod
    def test(checkpoint: str, data_path: str) -> float:
        initialize_logger()
        model = PyTorchTemplate._load_model(checkpoint)
        to_tensor = ToTensor()  # TODO: update transformations

        features = {'features': np.load(data_path).astype(np.float32)}  # TODO: update data loading
        features['features'] = to_tensor(features)
        prediction = model.predict(features)
        return prediction
