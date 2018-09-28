#!/usr/bin/env python3.6

# TODO: delete this template-specific file

import os

import numpy as np
import pandas as pd

FEATURE_SIZE = 5
WEIGHTS = np.array([5, 1, 3, 2, 4])
BIAS = 2
TRAINING_SET_SIZE = 80
VALIDATION_SET_SIZE = 20
TEST_SET_SIZE = 20

if __name__ == '__main__':
    os.makedirs(os.path.join('data', 'train'), exist_ok=True)
    os.makedirs(os.path.join('data', 'dev'), exist_ok=True)
    os.makedirs(os.path.join('data', 'test'), exist_ok=True)

    training_set = np.random.rand(TRAINING_SET_SIZE, FEATURE_SIZE)
    validation_set = np.random.rand(VALIDATION_SET_SIZE, FEATURE_SIZE)
    test_set = np.random.rand(TEST_SET_SIZE, FEATURE_SIZE)

    training_targets = training_set @ WEIGHTS + BIAS
    validation_targets = validation_set @ WEIGHTS + BIAS

    dataframe = pd.DataFrame(columns=['target'])
    dataframe.index.name = 'filepath'

    for i, features in enumerate(training_set):
        filename = os.path.join('train', 'train_{:03d}.npy'.format(i))
        np.save(os.path.join('data', filename), features)
        dataframe.loc[filename] = training_targets[i]

    for i, features in enumerate(validation_set):
        filename = os.path.join('dev', 'dev_{:03d}.npy'.format(i))
        np.save(os.path.join('data', filename), features)
        dataframe.loc[filename] = validation_targets[i]

    for i, features in enumerate(test_set):
        np.save(os.path.join('data', 'test', 'test_{:03d}.npy'.format(i)), features)

    dataframe.to_csv(os.path.join('data', 'targets.csv'), float_format='%.3f')
