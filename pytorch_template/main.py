import logging
import os
import pickle
import platform
import shutil
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ingestion.datasets import IngestDataset, NpyDataset
from .ingestion.transforms import ToTensor
from .models.linear import LinearRegression
from .models.model import Model

if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = os.cpu_count()


# TODO: update class name
class PyTorchTemplate:

    @staticmethod
    def _set_logger(silent: bool = False,
                    debug: bool = False) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if debug else logging.WARNING if silent else logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s | %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    @staticmethod
    def _create_working_env(output_dir: str) -> str:
        working_env = Path(output_dir).joinpath('runs', str(int(time.time())))
        Path.mkdir(working_env.joinpath('checkpoints'))
        Path.mkdir(working_env.joinpath('logs'))

        logger = logging.getLogger()
        file_handler = logging.FileHandler(
            working_env.joinpath('logs', 'train_info.log'))
        logger.addHandler(file_handler)

        return working_env

    @staticmethod
    def _load_model(checkpoint: str) -> Model:
        with open(Path(checkpoint).parent.joinpath('hyperparams.pkl'), 'rb') as f:
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
    def ingest(root_dir: str,
               split: str) -> str:
        PyTorchTemplate._set_logger()

        # TODO: add transformations

        dataset = IngestDataset(root_dir, split, 'targets.csv')
        loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])

        # TODO: update path
        output_dir = Path(root_dir).joinpath('npy', Path(split))
        if Path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir()

        for sample in tqdm(loader, desc='Writing {} feature files'.format(split)):
            output_path = output_dir.joinpath('{}.npy'.format(sample['filename']))
            np.save(output_path, np.array([sample['features'], sample['target']]))

        #  TODO: remove metadata file if not needed (as here)
        metadata_path = Path(root_dir).joinpath('npy', split, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({'num_files': len(dataset)}, f)
        return metadata_path

    @staticmethod
    def train(npy_dir: str,
              output_dir: str,
              batch_size: int,
              epochs: int,
              lr: float,
              silent: bool,
              debug: bool) -> str:
        PyTorchTemplate._set_logger(silent, debug)
        working_env = PyTorchTemplate._create_working_env(output_dir)

        logging.info('Batch size: {}'.format(batch_size))
        logging.info('Learning rate: {}'.format(lr))

        to_tensor = ToTensor()  # TODO: update transformations

        train_dataset = NpyDataset(npy_dir, 'train', transform=to_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        if Path(npy_dir).joinpath('dev').is_dir():
            dev_dataset = NpyDataset(npy_dir, 'dev', transform=to_tensor)
            dev_loader = DataLoader(
                dev_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
        else:
            dev_loader = None

        module = LinearRegression(train_dataset.features_shape[-1])
        model = Model(module)
        best_checkpoint = model.fit(
            working_env, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

    @staticmethod
    def restore(npy_dir: str,
                checkpoint: str,
                output_dir: str,
                batch_size: int,
                epochs: int,
                lr: float,
                silent: bool,
                debug: bool) -> str:
        PyTorchTemplate._set_logger(silent, debug)
        working_env = PyTorchTemplate._create_working_env(output_dir)

        logging.info('Checkpoint: {}'.format(checkpoint))
        logging.info('Batch size: {}'.format(batch_size))
        logging.info('Learning rate: {}'.format(lr))

        to_tensor = ToTensor()  # TODO: update transformations

        train_dataset = NpyDataset(npy_dir, 'train', transform=to_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        if Path(npy_dir).joinpath('dev').is_dir():
            dev_dataset = NpyDataset(npy_dir, 'dev', transform=to_tensor)
            dev_loader = DataLoader(
                dev_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
        else:
            dev_loader = None

        model = PyTorchTemplate._load_model(checkpoint)
        best_checkpoint = model.fit(
            working_env, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

    @staticmethod
    def evaluate(checkpoint: str,
                 npy_dir: str,
                 batch_size: int) -> Tuple[float, float]:
        # TODO: update transformations
        dev_dataset = NpyDataset(npy_dir, 'dev', transform=ToTensor())
        dev_loader = DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        model = PyTorchTemplate._load_model(checkpoint)
        val_loss, val_metric = model.eval(dev_loader)
        return val_loss, val_metric

    @staticmethod
    def test(checkpoint: str,
             data_path: str) -> float:
        PyTorchTemplate._set_logger()
        model = PyTorchTemplate._load_model(checkpoint)
        to_tensor = ToTensor()  # TODO: update transformations

        features = {'features': np.load(data_path).astype(np.float32)}  # TODO: update data loading
        features = to_tensor(features)
        prediction = model.predict(features)
        return prediction
