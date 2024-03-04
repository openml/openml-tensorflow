"""
Tensorflow image classification model example
==================

An example of a tensorflow network that classifies meta album images.
"""

import tensorflow

import openml
import openml_tensorflow

import os
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
############################################################################

############################################################################
openml.config.apikey = 'd7f058387fb3c8ba41e1ae61ebd999a0'

############################################################################

openml_tensorflow.config.epoch = 1 #  small epoch for test runs

IMG_SIZE = (128, 128)
IMG_SHAPE = IMG_SIZE + (3,)
base_learning_rate = 0.0001

datagen = ImageDataGenerator()
openml_tensorflow.config.datagen = datagen
openml_tensorflow.config.dir = openml.config.get_cache_directory()+'/datasets/44312/PNU_Micro/images/'
openml_tensorflow.config.x_col = "FILE_NAME"
openml_tensorflow.config.y_col = 'encoded_labels'
openml_tensorflow.config.datagen = datagen
openml_tensorflow.config.batch_size = 32
openml_tensorflow.config.class_mode = "categorical"

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# Example tensorflow image classification model. You can do better :)
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(19, activation='softmax'))  # Adjust output size
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

############################################################################

# Download the OpenML task for the Meta_Album_PNU_Micro dataset.
task = openml.tasks.get_task(361987)

# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################

# Visualize model in netron
import netron
import tf2onnx

# Convert the TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save the ONNX model to a file
onnx_file_path = "model.onnx"
with open(onnx_file_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Visualize the ONNX model using Netron
netron.start(onnx_file_path, browse=False)