# PyTorch Template

Code and documentation template for PyTorch research projects.
This repository is intended to be cloned at the beginning of any
new research deep learning project based on PyTorch.

## Getting Started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.

### Prerequisites

In order to run the code you need to have Python 3.6 installed.

### Installing

You can install the package on MacOS/Linux with the following commands:

```
python3.6 setup.py sdist
python3.6 setup.py bdist_wheel
pip3.6 install --no-index --find-links=dist pytorch_template -r requirements.txt
```

## Usage

A command line interface is available to easily interact with the package.

The available commands are:
- `ingest`: preprocess raw data and export it in a suitable format for model
training, necessary only if preprocessing is computationally expensive and
many transformations are required;
- `train`: train the deep learning model on ingested data
- `eval`: evaluate the model on ingested validation data
- `test`: produce model output on a single raw data sample.

### Ingest

TODO

### Train

TODO

### Eval

TODO

### Test

TODO

## Performances

TODO

## Deployment

TODO 

## License

This project is licensed under Proprietary License -
see the [LICENSE.md](LICENSE.md) file for details
