#!/usr/bin/env python3.6

import argparse
import sys

from pytorch_template.main import PyTorchTemplate  # TODO: update


class CLI:

    def __init__(self):
        #  TODO: update description and usage
        parser = argparse.ArgumentParser(
            description='Command line interface for PyTorch template',
            usage=('python3.6 cli.py <command> [<args>]\n'
                   '\n'
                   'ingest      Ingest data\n'
                   'train       Train the model\n'
                   'eval        Evaluate the model\n'
                   'test        Test the model\n'))
        parser.add_argument('command', type=str, help='Sub-command to run',
                            choices=['ingest', 'train', 'eval', 'test'])

        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace('-', '_')
        if not hasattr(self, command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def ingest() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Ingest data')
        #  TODO: update parameters and default values
        parser.add_argument('data_dir', type=str, help='Data directory')
        parser.add_argument('split', type=str, help='Split name',
                            choices=['train', 'dev', 'test'])
        parser.add_argument('--workers', type=int, default=1,
                            help='Number of workers')

        args = parser.parse_args(sys.argv[2:])
        PyTorchTemplate.ingest(args.data_dir, args.split, args.workers)

    @staticmethod
    def train() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Train the model')
        #  TODO: update parameters and default values
        parser.add_argument('npy_dir', type=str, help='Npy directory')
        parser.add_argument('--output-dir', type=str, help='Output directory',
                            default='./')
        parser.add_argument('--batch-size', type=int, default=20,
                            help='Batch size')
        parser.add_argument('--epochs', type=int, default=30,
                            help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='Initial learning rate')
        parser.add_argument('--workers', type=int, default=4,
                            help='Number of workers')

        args = parser.parse_args(sys.argv[2:])
        PyTorchTemplate.train(args.npy_dir, args.output_dir, args.batch_size,
                              args.epochs, args.lr, args.workers)

    @staticmethod
    def eval() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Evaluate the model')
        #  TODO: update parameters and default values
        parser.add_argument('model-dir', type=str, help='Model directory')
        parser.add_argument('npy-dir', type=str, help='Npy directory')
        parser.add_argument('--workers', type=int, default=1,
                            help='Number of workers')

        args = parser.parse_args(sys.argv[2:])
        PyTorchTemplate.evaluate(args.model_dir, args.npy_dir, args.workers)

    @staticmethod
    def test() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Test the model')
        #  TODO: update parameters and default values
        parser.add_argument('model-dir', type=str, help='Model directory')
        parser.add_argument('data-dir', type=str, help='Data directory')

        args = parser.parse_args(sys.argv[2:])
        PyTorchTemplate.test(args.model_dir, args.data_dir)


if __name__ == '__main__':
    CLI()
