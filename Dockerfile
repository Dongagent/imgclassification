FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="Dongagent <mlds.yang@gmail.com>"

# setup for the timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
 && apt-get install -y \
    sudo \
    git \
    wget \
    curl \
    vim \
    unzip \
    software-properties-common

# Python common packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    opencv-python \
    scipy \
    scikit-learn \
    pycaret \
    matplotlib \
    seaborn \
    plotly \
    tqdm \ 
    gdown

# For opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set workdir
WORKDIR /home/src

# Add files to docker image
ADD . /home/src
# Download dataset using gdown
RUN gdown https://drive.google.com/uc?id=11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E
# Download checkpoint using gdown
RUN gdown https://drive.google.com/uc?id=12yFdECJVeaBgQaw2lw4O0EbK7gOQndKn
RUN mkdir ckpt
RUN mv model_epoch56.pth ckpt/
# unzip dataset
RUN unzip -q screws.zip