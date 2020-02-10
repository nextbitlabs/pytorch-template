import torch
import torch.nn as nn


def initialize_weights(m: nn.Module) -> None:
    # TODO: update initialization (with gain)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class LinearRegression(nn.Module):
    def __init__(self, features_size: int):
        super(LinearRegression, self).__init__()
        # TODO: update model layers
        self.fc = nn.Linear(features_size, 1)
        self.hyperparams = {
            'module_name': 'LinearRegression',
            'features_size': features_size,
        }
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # noinspection PyUnresolvedReferences
        prediction = torch.squeeze(self.fc(x))
        return prediction
