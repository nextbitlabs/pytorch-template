import logging
import os
import pickle
from typing import Optional, Tuple, Dict

import tensorboardX
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.monitors import Monitor


class Model:
    SUMMARY_STEPS = 10  # TODO: update

    def __init__(self,
                 module: nn.Module):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module = module.to(self.device)
        self.criterion = nn.MSELoss()  # TODO: update

    def fit(self,
            working_env: str,
            loader: DataLoader,
            epochs: int,
            lr: float,
            dev_loader: Optional[DataLoader] = None) -> str:
        writer = tensorboardX.SummaryWriter(os.path.join(working_env, 'logs'))
        with open(os.path.join(working_env, 'checkpoints', 'hyperparams.pkl'), 'wb') as f:
            pickle.dump(self.module.hyperparams, f)

        validation = dev_loader is not None

        # TODO: update optimizer
        optimizer = torch.optim.SGD(self.module.parameters(), lr=lr,
                                    momentum=0.9, nesterov=True)
        # TODO: update scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.8)

        total_step = 0
        loss_monitor = Monitor()
        log_string = 'Epoch {:d} - Loss: {:.4f}'
        val_log_string = 'Validation after epoch {:d} - Loss: {:.4f} - L1: {:.4f}'
        best_checkpoint = ''
        best_val_metric = float('inf')  # TODO:update lower/upper bound
        for epoch in range(epochs):
            self.module.train()
            loss_monitor.reset()
            scheduler.step()
            for samples in tqdm(loader, desc='Epoch {}'.format(epoch)):
                optimizer.zero_grad()

                inputs = samples['features'].to(self.device)
                targets = samples['target'].to(self.device)
                predictions = self.module(inputs)
                loss = self.criterion(predictions, targets)

                loss.backward()
                optimizer.step()

                total_step += 1
                loss_monitor.update(loss)
                if total_step % self.SUMMARY_STEPS == 0:
                    writer.add_scalar('loss', loss_monitor.value, total_step)

            logging.info(log_string.format(epoch, loss_monitor.value))
            writer.add_scalar('lr', scheduler.get_lr()[0], total_step)

            if validation:
                val_loss, val_metric = self.eval(dev_loader)
                writer.add_scalar('val_loss', val_loss, total_step)
                writer.add_scalar('val_metric', val_metric, total_step)
                logging.info(val_log_string.format(epoch, val_loss, val_metric))
                checkpoint_filename = 'model-{:03d}_{:.3f}.ckpt'.format(
                    epoch, val_metric)
                if val_metric < best_val_metric:  # TODO: update inequality
                    best_checkpoint = checkpoint_filename
                    best_val_metric = val_metric
            else:
                checkpoint_filename = 'model-{:03d}.ckpt'.format(epoch)
                best_checkpoint = checkpoint_filename

            checkpoint_filepath = os.path.join(
                working_env, 'checkpoints', checkpoint_filename)
            torch.save(self.module.state_dict(), checkpoint_filepath)

        writer.close()
        best_checkpoint = os.path.join(
            working_env, 'checkpoints', best_checkpoint)
        return best_checkpoint

    def eval(self,
             dev_loader: DataLoader) -> Tuple[float, float]:
        val_loss_monitor = Monitor()
        val_metric_monitor = Monitor()
        metric = nn.L1Loss()  # TODO: update metrics

        self.module.eval()
        with torch.no_grad():
            for samples in dev_loader:
                inputs = samples['features'].to(self.device)
                targets = samples['target'].to(self.device)
                predictions = self.module(inputs)
                loss = self.criterion(predictions, targets)
                l1loss = metric(predictions, targets)

                val_loss_monitor.update(loss)
                val_metric_monitor.update(l1loss)

        return val_loss_monitor.value, val_metric_monitor.value

    def predict(self,
                sample: Dict[str, torch.Tensor]) -> float:
        features = sample['features'].to(self.device)
        self.module.eval()
        with torch.no_grad():
            output = self.module(features).item()  # TODO: update if output is a vector
        return output
