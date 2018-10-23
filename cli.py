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
                   'restore     Restore training from a checkpoint\n'
                   'eval        Evaluate the model\n'
                   'test        Test the model\n'))
        parser.add_argument('command', type=str, help='Sub-command to run',
                            choices=('ingest', 'train', 'restore', 'eval', 'test'))

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

        args = parser.parse_args(sys.argv[2:])
        metadata_path = PyTorchTemplate.ingest(args.data_dir, args.split)
        print('Metadata saved at {}'.format(metadata_path))

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
        parser.add_argument('--epochs', type=int, default=40,
                            help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='Initial learning rate')
        parser.add_argument('-s', '--silent', action='store_true',
                            help='Less verbose logging')
        parser.add_argument('--debug', action='store_true',
                            help='Log everything for debugging purpose')

        args = parser.parse_args(sys.argv[2:])
        best_checkpoint = PyTorchTemplate.train(
            args.npy_dir, args.output_dir, args.batch_size, args.epochs, args.lr,
            args.silent, args.debug)
        print('Best checkpoint saved at {}'.format(best_checkpoint))

    @staticmethod
    def restore() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Restore training from a checkpoint')
        #  TODO: update parameters and default values
        parser.add_argument('npy_dir', type=str, help='Npy directory')
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument('--output-dir', type=str, help='Output directory',
                            default='./')
        parser.add_argument('--batch-size', type=int, default=20,
                            help='Batch size')
        parser.add_argument('--epochs', type=int, default=40,
                            help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='Initial learning rate')
        parser.add_argument('-s', '--silent', action='store_true',
                            help='Less verbose logging')
        parser.add_argument('--debug', action='store_true',
                            help='Log everything for debugging purpose')

        args = parser.parse_args(sys.argv[2:])
        best_checkpoint = PyTorchTemplate.restore(
            args.npy_dir, args.checkpoint, args.output_dir, args.batch_size,
            args.epochs, args.lr, args.silent, args.debug)
        print('Best checkpoint saved at {}'.format(best_checkpoint))

    @staticmethod
    def eval() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Evaluate the model')
        #  TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument('npy_dir', type=str, help='Npy directory')
        parser.add_argument('--batch-size', type=int, default=20,
                            help='Batch size')

        args = parser.parse_args(sys.argv[2:])
        val_loss, val_metric = PyTorchTemplate.evaluate(
            args.checkpoint, args.npy_dir, args.batch_size)
        val_log_string = 'Validation - Loss: {:.4f} - Metric: {:.4f}'
        print(val_log_string.format(val_loss, val_metric))

    @staticmethod
    def test() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(description='Test the model')
        #  TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument('data_path', type=str, help='Data file path')

        args = parser.parse_args(sys.argv[2:])
        prediction = PyTorchTemplate.test(args.checkpoint, args.data_path)
        print('Output: {:.4f}'.format(prediction))


if __name__ == '__main__':
    CLI()
