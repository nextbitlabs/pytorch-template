import argparse
import json
import multiprocessing
import pathlib
import sys

import torch
import torch.utils.data

from pytorch_template.ingestion import IngestDataset
from pytorch_template.ingestion import ToTensor


def compute_statistics(root_dir: str) -> None:
    # TODO: update transform
    dataset = IngestDataset(
        root_dir=root_dir, split='train', transform=ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=20, num_workers=multiprocessing.cpu_count()
    )

    n = 0
    channels = dataset[0]['features'].size(-1)
    mean = torch.zeros(channels, dtype=torch.float32)
    m2 = torch.zeros(channels, dtype=torch.float32)

    for sample in loader:
        features = sample['features']
        # Here mean and standard deviation are computed via Chan et. al. algorithm
        # (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
        count = features.size(0)
        avg = features.mean(dim=0)
        var = features.var(dim=0)
        delta = avg - mean
        mean = (mean * n + features.sum(dim=0)) / (n + count)
        m2 += var * (count - 1) + delta.pow(2) * n * count / (n + count)
        n += count

    std = torch.sqrt(m2 / (n - 1))
    statistics = {'mean': mean.tolist(), 'std': std.tolist()}
    print(statistics)

    with open(pathlib.Path(root_dir) / 'statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect the mean and standard deviation for input data',
        usage='python3.7 dataset_statistics.py root-dir',
    )
    parser.add_argument(
        'root_dir', metavar='root-dir', type=str, help='Path to data root directory.'
    )
    args = parser.parse_args(sys.argv[1:])
    compute_statistics(args.root_dir)
