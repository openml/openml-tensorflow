# OpenML-Tensorflow container

The docker container has the latest version of OpenML-Tensorflow downloaded and pre-installed. It can be used to run TensorFlow Deep Learning analysis on OpenML datasets. 
This document contains information about:

[Usage](#usage): how to use the image 
[Using Locally Stored Datasets](#using-locally-stored-datasets-optional): mounting datasets from the local cache
[Environment Variables](#environment-variable): setting the cache directory path

## Usage

These are the steps to use the image:

1. Pull the docker image 
```
docker pull taniyadas/openml-tensorflow:latest
```
2. If you want to run a local script, it needs to be mounted first. Mount it into the 'app' folder:
```text
docker run -it -v PATH/TO/CODE_FOLDER:/app taniyadas/openml-tensorflow /bin/bash
```
You can also mount multiple directories into the container (such as your code file directory and dataset directory ) using:
```text
docker run -t -i -v PATH/TO/CODE_FOLDER:/app -v PATH/TO/DATASET_FOLDER:/app/dataset taniyadas/openml-tensorflow /bin/bash
```
3. Please make sure to give the correct path to the dataset. For example, 
```text
openml_tensorflow.config.dir = 'dataset/Images'
```
4. Run your code scripts using:
```text
 python my_code.py
```

## Using Locally Stored Datasets [Optional]

If you don't want to download the dataset each time you run your script, you can mount your dataset saved in your local cache directory to the container. 

### Example Usage

1. Mount the dataset to the 'app/dataset' folder

```
docker run -t -i -v PATH/TO/CODE_FOLDER:/app -v PATH/TO/DATASET_FOLDER:/app/dataset taniyadas/openml-tensorflow /bin/bash
```

2. Set correct path to the dataset.  
```text
openml_tensorflow.config.dir = '/app/dataset/Images'
``` 

## Environment Variable 

You can configure the cache directory to control where 'OpenML' datasets are downloaded and cached.

```
cache_dir = "/app/.openml"
openml.config.set_root_cache_directory(cache_dir)
```