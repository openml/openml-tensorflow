# Tensorflow extension for OpenML python

Tensorflow extension for [openml-python API](https://github.com/openml/openml-python). This library provides a simple way to run your Tensorflow models on OpenML tasks. 

#### Installation Instructions:

`pip install openml-tensorflow`

PyPi link https://pypi.org/project/openml-tensorflow/

#### Usage
Import openML libraries
```python
import openml
import openml_tensorflow
from tensorflow.keras import layers, models

```
Create  and compile a tensorflow model
```python

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=IMG_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(19, activation='softmax'))  
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['AUC'])

# We will compile using the Adam optimizer while targeting accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['AUC'])
```
Download the task from openML and run the model on task.
```python
task = openml.tasks.get_task(362071)
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()
print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

```

Note: The input layer of the network should be compatible with OpenML data output shape. Please check [examples](/docs/Examples/) for more information.


Additionally, if you want to publish the run with onnx file, then you must call ```openml_tensorflow.add_onnx_to_run()``` immediately before ```run.publish()```. 

```python
run = openml_tensorflow.add_onnx_to_run(run)
```

#### Using docker image

The docker container has the latest version of [OpenML-Tensorflow](https://github.com/openml/openml-tensorflow) downloaded and pre-installed. It can be used to run TensorFlow Deep Learning analysis on OpenML datasets. 
See [docker](./Docker%20reference/Docker.md).


This library is currently under development, please report any bugs or feature request in issues section.