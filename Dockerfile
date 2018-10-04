FROM nvcr.io/nvidia/pytorch:18.09-py3

RUN apt-get -y update
RUN apt-get -y install screen htop
RUN pip install --upgrade pip
RUN pip install gpustat
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pandas
RUN pip3 install --upgrade tensorflow
RUN pip3 install --upgrade tensorboard
RUN pip3 install --upgrade tensorboardX
RUN pip3 install --upgrade tqdm

EXPOSE 6006