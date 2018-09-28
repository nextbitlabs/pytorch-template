import logging
import os
import pickle
from typing import Optional

import tensorboardX
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.monitors import Monitor


class Model:

    def __init__(self,
                 working_env: str,
                 module: nn.Module):
        self.working_env = working_env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.module = module.to(self.device)

    def fit(self,
            loader: DataLoader,
            epochs: int,
            lr: float,
            dev_loader: Optional[DataLoader] = None):
        summary_steps = 10  # TODO: update
        writer = tensorboardX.SummaryWriter(os.path.join(self.working_env, 'logs'))
        with open(os.path.join(self.working_env, 'model_args.pkl'), 'wb') as f:
            pickle.dump(self.module.hyperparams, f)

        validation = dev_loader is not None

        criterion = nn.MSELoss()  # TODO: update loss
        # TODO: update optimizer
        optimizer = torch.optim.SGD(self.module.parameters(), lr=lr,
                                    momentum=0.9, nesterov=True)
        # TODO: update scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.8)

        total_step = 0
        loss_monitor = Monitor()
        log_string = 'Epoch {:d} - Loss: {:.4f}'

        val_loss_monitor = Monitor() if validation else None
        val_metric_monitor = Monitor() if validation else None
        val_log_string = 'Validation after epoch {:d} - Loss: {:.4f} - L1: {:.4f}'
        for epoch in range(epochs):
            self.module.train()
            loss_monitor.reset()
            scheduler.step()
            for features, targets in loader:
                optimizer.zero_grad()

                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = self.module(features)
                loss = criterion(predictions, targets)

                loss.backward()
                optimizer.step()

                total_step += 1
                loss_monitor.update(loss, targets)
                if total_step % summary_steps == 0:
                    writer.add_scalar('loss', loss_monitor.value, total_step)

            logging.info(log_string.format(epoch, loss_monitor.value))
            writer.add_scalar('lr', scheduler.get_lr()[0], total_step)

            if validation:
                metric = nn.L1Loss()  # TODO: update metrics

                with torch.no_grad():
                    self.module.eval()
                    val_loss_monitor.reset()
                    val_metric_monitor.reset()
                    for features, targets in dev_loader:
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                        predictions = self.module(features)
                        loss = criterion(predictions, targets)
                        l1loss = metric(predictions, targets)

                        val_loss_monitor.update(loss, targets)
                        val_metric_monitor.update(l1loss, targets)

                writer.add_scalar('val_loss', val_loss_monitor.value, total_step)
                writer.add_scalar('val_metric', val_metric_monitor.value, total_step)
                logging.info(val_log_string.format(
                    epoch, val_loss_monitor.value, val_metric_monitor.value))

                checkpoint_filename = 'model-{:03d}_{:.3f}.ckpt'.format(
                    epoch, val_metric_monitor.value)
            else:
                checkpoint_filename = 'model-{:03d}.ckpt'.format(epoch)

            checkpoint_filepath = os.path.join(
                self.working_env, 'checkpoints', checkpoint_filename)
            torch.save(self.module.state_dict(), checkpoint_filepath)

        writer.close()
