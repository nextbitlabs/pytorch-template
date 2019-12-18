#!/usr/bin/env python3.6

# TODO: delete this template-specific file

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SIZE = 5
WEIGHTS = np.array([5, 1, 3, 2, 4])
BIAS = 2
TRAINING_SET_SIZE = 80
VALIDATION_SET_SIZE = 20
TEST_SET_SIZE = 20

if __name__ == '__main__':

    Path('data').joinpath('train').mkdir(parents=True, exist_ok=True)
    Path('data').joinpath('dev').mkdir(parents=True, exist_ok=True)
    Path('data').joinpath('test').mkdir(parents=True, exist_ok=True)

    training_set = np.random.rand(TRAINING_SET_SIZE, FEATURE_SIZE)
    validation_set = np.random.rand(VALIDATION_SET_SIZE, FEATURE_SIZE)
    test_set = np.random.rand(TEST_SET_SIZE, FEATURE_SIZE)

    training_targets = training_set @ WEIGHTS + BIAS
    validation_targets = validation_set @ WEIGHTS + BIAS

    dataframe = pd.DataFrame(columns=['target'])
    dataframe.index.name = 'filepath'

    for i, features in enumerate(training_set):
        filename = Path('train').joinpath('train_{:03d}.npy'.format(i))
        np.save(Path('data').joinpath(filename), features)
        dataframe.loc[filename] = training_targets[i]

    for i, features in enumerate(validation_set):
        filename = Path('dev').joinpath('dev_{:03d}.npy'.format(i))
        np.save(Path('data').joinpath(filename), features)
        dataframe.loc[filename] = validation_targets[i]

    for i, features in enumerate(test_set):
        np.save(Path('data').joinpath('test', 'test_{:03d}.npy'.format(i)), features)

    dataframe.to_csv(Path('data').joinpath('targets.csv'), float_format='%.3f')
