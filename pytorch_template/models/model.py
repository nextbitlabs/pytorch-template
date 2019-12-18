import logging
import os
import pickle
from typing import Optional, Tuple, Dict

import tensorboardX
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.monitor import Monitor


class Model:
    SUMMARY_STEPS = 10  # TODO: update

    @staticmethod
    def _get_optimizer_info(optimizer: torch.optim.Optimizer) -> str:
        info = type(optimizer).__name__
        for attr in optimizer.defaults:
            info = info + " | {}: {}".format(attr, optimizer.defaults[attr])
        return info

    # noinspection PyProtectedMember
    @staticmethod
    def _get_scheduler_info(scheduler: torch.optim.lr_scheduler._LRScheduler) -> str:
        attr_to_exclude = ('optimizer', 'is_better')
        info = type(scheduler).__name__
        for attr in scheduler.__dict__:
            if hasattr(scheduler, attr) and attr not in attr_to_exclude:
                info = info + " | {}: {}".format(attr, scheduler.__dict__[attr])
        return info

    def __init__(self,
                 module: nn.Module):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module = module.to(device)
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
                                    momentum=0.9, nesterov=True, weight_decay=1e-4)
        logging.info('Optimizer: {}'.format(self._get_optimizer_info(optimizer)))

        # TODO: update scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.8)
        logging.info('Scheduler: {}'.format(self._get_scheduler_info(scheduler)))

        total_step = 0
        val_log_string = 'Validation after epoch {:d} - Loss: {:.4f} - L1: {:.4f}'
        best_checkpoint = ''
        best_val_metric = float('inf')  # TODO:update lower/upper bound
        for epoch in range(epochs):
            self.module.train()
            scheduler.step()
            for samples in tqdm(loader, desc='Epoch {}'.format(epoch)):
                optimizer.zero_grad()

                inputs = samples['features']
                targets = samples['target']
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                predictions = self.module(inputs)
                loss = self.criterion(predictions, targets)

                loss.backward()
                optimizer.step()

                total_step += 1
                if total_step % self.SUMMARY_STEPS == 0:
                    writer.add_scalar('loss', loss.item(), total_step)

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
                inputs = samples['features']
                targets = samples['target']
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                predictions = self.module(inputs)
                loss = self.criterion(predictions, targets)
                l1loss = metric(predictions, targets)

                val_loss_monitor.update(loss)
                val_metric_monitor.update(l1loss)

        return val_loss_monitor.value, val_metric_monitor.value

    def predict(self,
                sample: Dict[str, torch.Tensor]) -> float:
        features = sample['features']
        if torch.cuda.is_available():
            features = features.cuda(non_blocking=True)

        self.module.eval()
        with torch.no_grad():
            output = self.module(features).item()  # TODO: update if output is a vector
        return output
