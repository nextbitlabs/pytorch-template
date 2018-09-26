import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self,
                 features_size: int):
        super(LinearRegression, self).__init__()
        # TODO: update model layers
        self.fc = nn.Linear(features_size, 1)

        # TODO: update initialization (with gain)
        nn.init.xavier_normal_(self.fc.weight)

        self.hyperparams = {
            'features_size': features_size
        }

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        prediction = torch.squeeze(self.fc(x))
        return prediction
