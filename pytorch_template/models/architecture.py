import logging
import os
import pickle
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Architecture:

    def __init__(self,
                 working_env: str,
                 model: nn.Module):
        self.working_env = working_env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def fit(self,
            loader: DataLoader,
            epochs: int,
            lr: float,
            dev_loader: Optional[DataLoader] = None):
        with open(os.path.join(self.working_env, 'model_args.pkl'), 'wb') as f:
            pickle.dump(self.model.hyperparams, f)

        validation = dev_loader is not None
        # TODO: update loss
        criterion = nn.MSELoss()
        # TODO: update optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                    momentum=0.9, nesterov=True)
        # TODO: update scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.8)

        log_string = 'Epoch {:d} - Step {:d} - Loss: {:.4f}'
        val_log_string = 'Validation after epoch {:d} - Loss: {:.4f} - L1: {:.4f}'

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            for step, (features, targets) in enumerate(loader, start=1):
                optimizer.zero_grad()

                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(features)
                loss = criterion(predictions, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.data
                logging.info(log_string.format(epoch, step, total_loss / step))

            if validation:
                metric = nn.L1Loss()

                total_loss = 0
                total_metric = 0
                with torch.no_grad():
                    self.model.eval()
                    for features, targets in dev_loader:
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                        predictions = self.model(features)
                        loss = criterion(predictions, targets)
                        l1loss = metric(predictions, targets)

                        total_loss += loss.data
                        total_metric += l1loss.data

                mean_loss = total_loss / len(dev_loader.dataset)
                mean_metric = total_metric / len(dev_loader.dataset)
                logging.info(val_log_string.format(epoch, mean_loss, mean_metric))

                checkpoint_filename = 'model-{}_{:.3f}.ckpt'.format(
                    epoch, total_metric / len(dev_loader.dataset))
            else:
                checkpoint_filename = 'model-{}.ckpt'.format(epoch)

            checkpoint_filepath = os.path.join(
                self.working_env, 'checkpoints', checkpoint_filename)
            print(self.model.state_dict())
            torch.save(self.model.state_dict(), checkpoint_filepath)
