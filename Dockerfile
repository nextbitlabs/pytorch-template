FROM nvcr.io/nvidia/pytorch:18.09-py3

RUN apt-get -y update
RUN apt-get -y install screen htop
RUN pip install --upgrade pip
RUN pip install gpustat
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pandas
RUN pip3 install --upgrade tensorboard
RUN pip3 install --upgrade tensorboardX
RUN pip3 install --upgrade tqdm

COPY . /workspace

WORKDIR /workspace

RUN python3 setup.py sdist && \
    python3 setup.py bdist_wheel && \
    pip3 install --no-index --find-links=dist pytorch_template

EXPOSE 6006