#!/usr/bin/env python3

import argparse
import sys

from pytorch_template.app import PyTorchTemplate  # TODO: update


class CLI:
    def __init__(self):
        #  TODO: update description and usage
        parser = argparse.ArgumentParser(
            description='Command line interface for PyTorch template',
            usage=(
                'python3 -m pytorch_template <command> [<args>]\n'
                '\n'
                'ingest      Ingest data\n'
                'train       Train the model\n'
                'restore     Restore training from a checkpoint\n'
                'eval        Evaluate the model\n'
                'test        Test the model\n'
            ),
        )
        parser.add_argument(
            'command',
            type=str,
            help='Sub-command to run',
            choices=('ingest', 'train', 'restore', 'eval', 'test'),
        )

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
        parser = argparse.ArgumentParser(
            description='Ingest data',
            usage='python3 -m pytorch_template ingest data-dir split',
        )
        #  TODO: update parameters and default values
        parser.add_argument(
            'data_dir', metavar='data-dir', type=str, help='Data directory'
        )
        parser.add_argument(
            'split', type=str, help='Split name', choices=('train', 'dev', 'test')
        )

        args = parser.parse_args(sys.argv[2:])
        PyTorchTemplate.ingest(args.data_dir, args.split)
        print(f'Ingestion completed')

    @staticmethod
    def train() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Train the model',
            usage='python3 -m pytorch_template train tensor-dir '
            '[--output-dir OUTPUT-DIR --batch-size BATCH-SIZE '
            '--epochs EPOCHS --lr LR]',
        )
        #  TODO: update parameters and default values
        parser.add_argument(
            'tensor_dir', metavar='tensor-dir', type=str, help='Tensors directory'
        )
        parser.add_argument(
            '--output-dir', type=str, help='Output directory', default='./runs'
        )
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
        parser.add_argument(
            '--lr', type=float, default=0.1, help='Initial learning rate'
        )

        args = parser.parse_args(sys.argv[2:])
        best_checkpoint = PyTorchTemplate.train(
            args.tensor_dir, args.output_dir, args.batch_size, args.epochs, args.lr
        )
        print(f'Best checkpoint saved at {best_checkpoint}')

    @staticmethod
    def restore() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Restore training from a checkpoint',
            usage='python3 -m pytorch_template restore tensor-dir '
            '[--output-dir OUTPUT-DIR --batch-size BATCH-SIZE '
            '--epochs EPOCHS --lr LR]',
        )
        #  TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            'tensor_dir', metavar='tensor-dir', type=str, help='Tensors directory'
        )
        parser.add_argument(
            '--output-dir', type=str, help='Output directory', default='./runs'
        )
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
        parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
        parser.add_argument(
            '--lr', type=float, default=0.1, help='Initial learning rate'
        )

        args = parser.parse_args(sys.argv[2:])
        best_checkpoint = PyTorchTemplate.restore(
            args.checkpoint,
            args.tensor_dir,
            args.output_dir,
            args.batch_size,
            args.epochs,
            args.lr,
        )
        print(f'Best checkpoint saved at {best_checkpoint}')

    @staticmethod
    def eval() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Evaluate the model',
            usage='python3 -m pytorch_template eval checkpoint tensor-dir '
            '[--batch-size BATCH-SIZE]',
        )
        #  TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            'tensor_dir', metavar='tensor-dir', type=str, help='Tensors directory'
        )
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size')

        args = parser.parse_args(sys.argv[2:])
        val_loss, val_metric = PyTorchTemplate.evaluate(
            args.checkpoint, args.tensor_dir, args.batch_size
        )
        print(f'Validation - Loss: {val_loss:.4f} - Metric: {val_metric:.4f}')

    @staticmethod
    def test() -> None:
        #  TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Test the model',
            usage='python3 -m pytorch_template test checkpoint data-path',
        )
        #  TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            'data_path', metavar='data-path', type=str, help='Data file path'
        )

        args = parser.parse_args(sys.argv[2:])
        prediction = PyTorchTemplate.test(args.checkpoint, args.data_path)
        print(f'Output: {prediction:.4f}')


if __name__ == '__main__':
    CLI()
