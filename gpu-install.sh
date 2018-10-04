#!/usr/bin/env bash

nvidia-docker build -t template-env .

nvidia-docker run \
	--name=template \
	--volume=$HOME/ebs:/ebs \
	--volume=$HOME/pytorch-template:/pytorch-template \
	-p 16006:6006 \
	-it template-env:latest /bin/bash

nvidia-docker start template
nvidia-docker exec -it template /bin/bash
