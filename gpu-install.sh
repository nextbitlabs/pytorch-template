#!/usr/bin/env bash

nvidia-docker build -t template-env . # TODO: update name

# TODO: update name and volume folders
nvidia-docker run \
	--name=template \
	--volume=$HOME/ebs:/ebs \
	--volume=$HOME/pytorch-template:/pytorch-template \
	-p 16006:6006 \
	-it template-env:latest /bin/bash

nvidia-docker start template # TODO: update name
nvidia-docker exec -it template /bin/bash # TODO: update name
