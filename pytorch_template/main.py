import json
import logging
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .ingestion.datasets import IngestDataset, TorchDataset
from .ingestion.transforms import Normalize, ToTensor
from .models.linear import LinearRegression
from .models.model import Model
from .utils.logger import initialize_logger


# TODO: update class name
class PyTorchTemplate:
    @staticmethod
    def _load_model(checkpoint: str) -> Model:
        with open(Path(checkpoint).parent.parent / 'hyperparams.json', 'r') as f:
            hyperparams = json.load(f)

        if LinearRegression.__name__ == hyperparams['module_name']:  # TODO: update module
            module_class = LinearRegression  # TODO: update module
        else:
            raise ValueError('Checkpoint of unsupported module')
        del hyperparams['module_name']

        module = module_class(**hyperparams)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        module.load_state_dict(torch.load(checkpoint, map_location=device))
        model = Model(module)
        return model

    @staticmethod
    def ingest(root_dir: str, split: str) -> None:
        initialize_logger()

        # TODO: update transformations
        # TODO: Compose is not needed in case of a single transform inside
        transform = Compose(
            [
                ToTensor(),
                # TODO: ToTensor is useless in this case, but preserved as an example.
                #  Check if you really need it
                # TODO: if you need normalization, replace values with statistics computed by
                #  dataset_statistics.py ; else remove it.
                Normalize(
                    mean=(0.502, 0.475, 0.475, 0.534, 0.493),
                    std=(0.283, 0.277, 0.281, 0.302, 0.306),
                ),
            ]
        )

        dataset = IngestDataset(root_dir, split, transform=transform)
        loader = DataLoader(dataset, batch_size=None, num_workers=multiprocessing.cpu_count())

        # TODO: update path
        output_dir = Path(root_dir) / 'tensors' / split
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)

        for sample in tqdm(loader, desc=f'Writing {split} feature files'):
            output_path = output_dir / f"{sample['filename']}.pt"
            torch.save(([sample['features'], sample['target']]), output_path)

    @staticmethod
    def train(tensor_dir: str, output_dir: str, batch_size: int, epochs: int, lr: float) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        train_dataset = TorchDataset(tensor_dir, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        if (Path(tensor_dir) / 'dev').is_dir():
            dev_dataset = TorchDataset(tensor_dir, 'dev')
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
            checkpoint: str, tensor_dir: str, output_dir: str, batch_size: int, epochs: int,
            lr: float
    ) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Checkpoint: {checkpoint}')
        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        train_dataset = TorchDataset(tensor_dir, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        if (Path(tensor_dir) / 'dev').is_dir():
            dev_dataset = TorchDataset(tensor_dir, 'dev')
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
    def evaluate(checkpoint: str, tensor_dir: str, batch_size: int) -> Tuple[float, float]:
        dev_dataset = TorchDataset(tensor_dir, 'dev')
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

        # TODO: update transformations to be coherent with what was used during training
        transform = Compose(
            [
                ToTensor(),
                # TODO: if you need normalization, replace values with statistics computed by
                #  dataset_statistics.py ; else remove it.
                Normalize(
                    mean=(0.502, 0.475, 0.475, 0.534, 0.493),
                    std=(0.283, 0.277, 0.281, 0.302, 0.306),
                ),
            ]
        )

        features = {'features': transform(torch.load(data_path))}  # TODO: update data loading
        prediction = model.predict(features)
        return prediction
