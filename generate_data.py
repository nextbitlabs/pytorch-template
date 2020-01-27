#!/usr/bin/env python3

# TODO: delete this template-specific file

from pathlib import Path

import pandas as pd
import torch

FEATURE_SIZE = 5
WEIGHTS = torch.tensor([5, 1, 3, 2, 4], dtype=torch.float)
BIAS = 2
TRAINING_SET_SIZE = 80
VALIDATION_SET_SIZE = 20
TEST_SET_SIZE = 20

if __name__ == '__main__':
    torch.manual_seed(0)

    (Path('data') / 'train').mkdir(parents=True, exist_ok=True)
    (Path('data') / 'dev').mkdir(parents=True, exist_ok=True)
    (Path('data') / 'test').mkdir(parents=True, exist_ok=True)

    training_set = torch.rand(TRAINING_SET_SIZE, FEATURE_SIZE)
    validation_set = torch.rand(VALIDATION_SET_SIZE, FEATURE_SIZE)
    test_set = torch.rand(TEST_SET_SIZE, FEATURE_SIZE)

    training_targets = training_set @ WEIGHTS + BIAS
    validation_targets = validation_set @ WEIGHTS + BIAS

    dataframe = pd.DataFrame(columns=['target'])
    dataframe.index.name = 'filepath'

    for i, features in enumerate(training_set):
        filename = Path('train') / f'train_{i:03d}.pt'
        torch.save(features, Path('data') / filename)
        dataframe.loc[filename] = training_targets[i].item()

    for i, features in enumerate(validation_set):
        filename = Path('dev') / f'dev_{i:03d}.pt'
        torch.save(features, Path('data') / filename)
        dataframe.loc[filename] = validation_targets[i].item()

    for i, features in enumerate(test_set):
        torch.save(features, Path('data') / 'test' / f'test_{i:03d}.pt')

    dataframe.to_csv(Path('data') / 'targets.csv', float_format='%.3f')
