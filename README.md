# PyTorch Template

Code and documentation template for PyTorch research projects.
This repository is intended to be cloned at the beginning of any
new research deep learning project based on PyTorch.

## Getting Started

These instructions will get you a copy of the project up and running
on your local machine for development and testing purposes.

The project folder, including also files excluded from git versioning,
has the following structure:

```
pytorch-template/                   [main folder]
│   .gitignore                      [files ignored by git]
│   .gitlab-ci.yml                  [GitLab CI/CD pipelines]
│   cli.py                          [package command-line interface]
│   LICENSE                         [code license]
│   README.md                       [this file]
│   requirements.txt                [package dependencies]
│   setup.py                        [package setup script]
│   setup.cfg                       [parameters for CI/CD]
│
├───data                            [data folder excluded from git tracking]
│   │   targets.csv                 [targets for train, dev and test data]
│   │
│   ├───train
│   │       ...
│   ├───dev
│   │       ...
│   └───test
│           ...
│
├───docs                            [documentation folder]
│       ...
│
└───pytorch_template                [package source code folder]
        ...
```

You should comply to this structure in all your projects,
in particular you should structure the `data` folder containing your dataset
according to the hierarchy shown. 

### Prerequisites

In order to run the code you need to have Python 3.6 installed.

### Installing

You can install the package on MacOS/Linux with the following commands:
```
git clone git@gitlab.com:nextbit/AI-research/pytorch-template.git
cd pytoorch-template
python3.6 setup.py sdist
python3.6 setup.py bdist_wheel
pip3.6 install --no-index --find-links=dist pytorch_template -r requirements.txt
```

Here data are synthetic so, in order to generate them run:
```
python3.6 generate_data.py
```

## Usage

A command line interface is available to easily interact with the package.
It is defined in the file `cli.py`.

To see more details about the command line interface
it is possible to show the help page using the command:
```
python3.6 cli.py --help
``` 

The available commands are:
- `ingest`: preprocess raw data and export it in a suitable format for model
training;
- `train`: train the deep learning model on ingested data;
- `eval`: evaluate the model on ingested validation data;
- `test`: produce model output on a single raw data sample.

Every command has its separate help page that can be visualized with
```
python3.6 cli.py <command> --help
```

### Command `ingest`

The ingestion phase is useful if preprocessing is computationally expensive and
many transformations are required. Here, for example, it is not really necessary
but it is included to show the code structure.

Optionally the `ingest` command can save some metadata about the dataset split,
like dataset global statistics computed during the full pass of the dataset,
in a pickle `.pkl` file and return the path of this file.

In some cases an additional `safe-ingest` can be used to check and assure labels 
coherence among the different dataset splits or to perform transformations
that depend on other splits. Here it is not needed because the
set of labels is not fixed since the example task is a regression.

#### Flow

```mermaid
graph TD;
    DataLoader-->|loops over|IngestDataset
    IngestDataset-->|selects a|sample
    sample-->|is normalized by|Normalize
    Normalize-->|and saved on disk by|ToFile
    ToFile-->decision{ingestion<br/>completed?}
    decision-->|no|DataLoader
    decision-->|yes|metadata
    
    style metadata stroke:#77d,stroke-width:2px
```

#### Examples

Only the training set and the development set have to be ingested
and that can be do with the following lines:
```
python3.6 cli.py ingest data/ train --workers 4
python3.6 cli.py ingest data/ dev --workers 4
```

For more details on the usage you can access the help page with the command
```
python3.6 cli.py ingest --help
```

### Command `train`

The training phase has always the same structure and the template is built
to keep all the tried models in files separated from the main training function.

#### Flow 

The flow refers to a single epoch inside the `fit` function of the `Model` class.

```mermaid
graph TD;
    DataLoader-->|loops over|NpyDataset
    NpyDataset-->|selects|samples
    samples-->|are composed by|features
    samples-->|are composed by|targets
    features-->|are analyzed by|LinearRegression1[LinearRegression]
    LinearRegression1-->|produces|predictions
    predictions-->|contribute to|MSELoss
    targets-->|contribute to|MSELoss
    MSELoss-->|updates|LinearRegression2[LinearRegression]
    LinearRegression2-->NpyDataset
    LinearRegression2-->|is validated on|L1Loss
    LinearRegression2-->|is saved on|checkpoint[checkpoint file]
```


#### Examples

The command has many optional parameters like `batch-size`, `epochs`, `lr`.

The most basic training can be performed using only default values
```
python3.6 cli.py train data/npy
```

An equivalent form of the previous command with all the default values
manually specified is:
```
python3.6 cli.py train data/npy \
    --output-dir . \
    --batch-size 20 \
    --epochs 40 \
    --lr 0.1 \
    --workers 4
```

For more details on the usage you can access the help page with the command
```
python3.6 cli.py train --help
```

### Command `eval`

TODO

### Command `test`

TODO

## Performances

TODO

## Deployment

TODO 

## License

This project is licensed under Proprietary License -
see the [LICENSE](LICENSE) file for details
