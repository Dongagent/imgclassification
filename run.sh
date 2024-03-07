#!/bin/bash

# exit when any command fails
set -e

# # Download dataset
# python download_dataset.py

# # unzip dataset
# unzip screws.zip

# train model
python train.py --epochs 60 --folder model1

# run inference
python inference.py