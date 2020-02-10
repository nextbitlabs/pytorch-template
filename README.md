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
│   cli.py                          [package command-line interface]
│   LICENSE                         [code license]
│   README.md                       [this file]
│   requirements.txt                [package dependencies]
│   setup.py                        [package setup script]
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
git clone https://github.com/nextbitlabs/pytorch-template.git
cd pytorch-template
python3 setup.py sdist
python3 setup.py bdist_wheel
pip3 install --no-index --find-links=dist pytorch_template -r requirements.txt
```

Here data are synthetic so, in order to generate them run:
```
python3 generate_data.py
```

## Usage

A command line interface is available to easily interact with the package.
It is defined in the file `cli.py`.

To see more details about the command line interface
it is possible to show the help page using the command:
```
python3 cli.py --help
``` 

The available commands are:
- `ingest`: preprocess raw data and export it in a suitable format for model
training;
- `train`: train the deep learning model on ingested data;
- `restore`: restore the training from a saved checkpoint;
- `eval`: evaluate the model on ingested validation data;
- `test`: produce model output on a single raw data sample.

Every command has its separate help page that can be visualized with
```
python3 cli.py <command> --help
```

### Command `ingest`

The ingestion phase is useful if preprocessing is computationally expensive and
many transformations are required. Here, for example, it is not really necessary
but it is included to show the code structure.

In some cases an additional `safe-ingest` can be used to check and assure labels 
coherence among the different dataset splits or to perform transformations
that depend on other splits. Here it is not needed because the
set of labels is not fixed since the example task is a regression.

#### Examples

Only the training set and the development set have to be ingested
and that can be do with the following lines:
```
python3 cli.py ingest data train
python3 cli.py ingest data dev
```

For more details on the usage you can access the help page with the command
```
python3 cli.py ingest --help
```

### Command `train`

The training phase has always the same structure and the template is built
to keep all the tried models in files separated from the main training function.

The path to the best weight checkpoint according to the metric is printed
to console at the end of the computation.

#### Examples

The command has many optional training-related parameters commonly tuned by the 
experimenter, like `batch-size`, `epochs`, `lr`. 
Logging options are also available: `silent` to log only warning 
messages, and `debug` for a more verbose logging. 

The most basic training can be performed specifying just the directory containing
the dataset, already split in `train` (compulsory) and `dev` (optional) folders
using the default values for the other parameters.
```
python3 cli.py train data/tensors
```

An equivalent form of the previous command with all the default values
manually specified is:
```
python3 cli.py train \
    data/tensors \
    --output-dir . \
    --batch-size 20 \
    --epochs 40 \
    --lr 0.1
```

For more details on the usage you can access the help page with the command
```
python3 cli.py train --help
```

### Command `restore`

When the model has not converged at the end of the training phase, it
can be useful to restore it from the last saved checkpoint and that is exactly
the role of this command.

#### Examples


The command has the same optional parameters of the `train` command.
It just has an additional compulsory parameter: the path to the checkpoint model
to be restored.

The most basic restored training can be performed specifying just the directory
containing the dataset, already split in `train` (compulsory) and `dev` (optional)
folders, and the checkpoint path using the default values for the other parameters.
```
python3 cli.py restore runs/<secfromepochs>/checkpoints/model-<epoch>-<metric>.ckpt data/tensors
```

An equivalent form of the previous command with all the default values
manually specified is:
```
python3 cli.py restore \
    runs/<secfromepochs>/checkpoints/model-<epoch>-<metric>.ckpt \
    data/tensors \
    --output-dir . \
    --batch-size 20 \
    --epochs 40 \
    --lr 0.1
```

For more details on the usage you can access the help page with the command
```
python3 cli.py restore --help
```

### Command `eval`

The `eval` command reproduces the validation performed at the end of every epoch during the training phase.
It is particularly useful when many datasets are available to evaluate the transfer learning performances.

#### Examples

The evaluation can be performed specifying just the model checkpoint
to be evaluated and the directory containing the dataset, provided of a `dev` sub-folders.
The batch size of evaluation batches can be manually specified otherwise its default
value is 20.

A full call to the command is:
```
python3 cli.py eval \
    runs/<secfromepochs>/checkpoints/model-<epoch>-<metric>.ckpt \
    data/tensors \
    --batch-size 20
```

For more details on the usage you can access the help page with the command
```
python3 cli.py eval --help
```

### Command `test`

The `test` command preforms the inference on a single file.

#### Examples

The test of the model is performed specifying the model checkpoint to be evaluated
and the path to a sample, for example:
```
python3 cli.py test \
    runs/<secfromepochs>/checkpoints/model-<epoch>-<metric>.ckpt \
    data/test/<sample>.pt
```

For more details on the usage you can access the help page with the command
```
python3 cli.py test --help
```

## Performances

The model converges to perfect predictions using default parameters.

## Deployment

The template can be deployed on an NGC optimized instance, here we list
the steps necessary to configure it on a AWS EC2 **g4dn.xlarge** instance
on the **NVIDIA Volta Deep Learning AMI** environment.

1. Log in via ssh following the instructions on the EC2 Management Dashboard.
2. Clone the repo `pytorch-template` in the home directory.
3. Download the most update PyTorch container running 
`docker pull nvcr.io/nvidia/pytorch:YY.MM-py3`
4. Create a container with
```
docker run --gpus all --name template -e HOME=$HOME -e USER=$USER \
    -v $HOME:$HOME -p 6006:6006 --shm-size 60G -it nvcr.io/nvidia/pytorch:YY.MM-py3
```
At the end of the procedure you will gain access to a terminal on a Docker
container configured to work on the GPU and you could simply run the commands
above leveraging the speed of parallel computing.

The `$HOME` directory on the Docker container is linked to the `$HOME` directory
of the host machine, so the repository can be found in the `$HOME`, similarly the
port 6006 used by TensorBoard is remapped from the container to the port 6006
of the host machine.

Useful commands to interact with the Docker container are:
- `docker start template`: start the container;
- `docker exec -it template bash`: open a terminal on the container;
- `docker stop template`: stop the container;
- `docker rm template`: remove the container.

In order to monitor training you can run the following commands from the container console:
- `watch -n 1 nvidia-smi` to monitor GPU usage;
- `tensorboard --logdir runs/<run_id> --bind_all` to start Tensorboard.

## Other 
The template also includes an implementation of a cool new optimizer,
[Ranger](https://medium.com/@lessw/2dc83f79a48d). 
Ranger uses the [Lookahead](https://arxiv.org/abs/1907.08610) optimization method 
together with the [RAdam](https://arxiv.org/abs/1908.03265) optimizer.
It is not used here, as it is way too slow for such a simple model,
but it reportedly performs better than other Adam variants on deeper models. 

You can use it simply by calling:
 ```
from pytorch_template.models.optimizer.ranger import Ranger

optimizer = Ranger(module.parameters())
```

If you want, you can specify many more hyper-parameters. 
If you use a learning rate scheduler, you should make sure that the learning rate remains
constant for a rather long time, in order to let RAdam start correctly and to take
advantage of LookAhead exploration.

An implementation of the [Mish](https://arxiv.org/abs/1908.08681) activation function is
also included. Mish seems to perform slightly better than ReLu when training deep models,
and works well in conjunction with Ranger. 

To use Mish, you just need to call it:
```
from pytorch_template.models.mish import Mish

mish = Mish()
```

## License

This project is licensed under Apache License 2.0,
see the [LICENSE](LICENSE) file for details.