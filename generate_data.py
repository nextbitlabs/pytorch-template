#!/usr/bin/env python3.6

# TODO: delete this template specific file

import os

import numpy as np

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

    training_targets = training_set @ WEIGHTS + BIAS + np.random.normal(TRAINING_SET_SIZE)
    validation_targets = validation_set @ WEIGHTS + BIAS + np.random.normal(VALIDATION_SET_SIZE)

    for i, sample in enumerate(zip(training_set, training_targets)):
        np.save(os.path.join('data', 'train', 'train_{:03d}.npy'.format(i)), sample)

    for i, sample in enumerate(zip(validation_set, validation_targets)):
        np.save(os.path.join('data', 'dev', 'dev_{:03d}.npy'.format(i)), sample)

    for i, sample in enumerate(test_set):
        np.save(os.path.join('data', 'test', 'test_{:03d}.npy'.format(i)), sample)
