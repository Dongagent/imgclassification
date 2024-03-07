# Assignment
[toc]
# Instructions
To ensure the environment is stable between different machines, I choose to use dockerfile for building my environment setup. 
FYI: I used one A100 40GB GPU for training.

Please make sure you have `docker` and `nvidia-docker` installed on your machine.

## Part 1: Environment setup
### First, build docker image using Dockerfile

You may run this command in the assignment folder
```
$ docker build -t dongsheng_yang_assignment .
```

It will **automatically setup** the necessary environment for this docker image. This process will take around **10 minutes**.


After loading the docker image, you can use `docker images | grep dongsheng_yang_assignment` to check the docker image is loaded well or not.

### Second, Run a nvidia docker container 

```
nvidia-docker run -it --name yds_assignment dongsheng_yang_assignment /bin/bash 
```

By the way, if you want to link a folder from your machine to the container, use `-v YOUR_ABSOLUTE_FOLDER_PATH:/home/src`

## Part 2: training and inference

### Files
In `/home/src`, you should see the files below.
```
.
|-- Dockerfile
|-- README.md
|-- archive
|   |-- license.txt
|   |-- test
|   `-- train
|-- datasets.py
|-- download_dataset.py
|-- inference.py
|-- model.py
|-- run.sh
|-- screws.zip
|-- test_data.json
`-- train.py
```

### Oneline command
If you need to valid from training to inference, use this oneline command.
```
bash run.sh
```

### Step by step command
Datasets are already downloaded and unzipped while building this docker image. Or you can use the `python download_dataset.py`.
#### Step 1 Training
```
python train.py --epochs 60 --folder model1
```
You will see the models in the path`./outputs/model1/`


#### Step 2 Inference
```
python inference.py
```
You will see analysis figures in `./analysis_figures/`
