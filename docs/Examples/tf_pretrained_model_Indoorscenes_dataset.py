"""
Tensorflow image classification using pre-trained model example II
==================

An example of a tensorflow network that classifies Indoor Scenes images using pre-trained transformer model.
Here some layers of the pre-trained model are trained while other layers are frozen. 
"""


import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import openml
import openml_tensorflow

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
############################################################################

############################################################################
openml.config.apikey = 'KEY'

############################################################################

openml_tensorflow.config.epoch = 1 #  small epoch for test runs

IMG_SIZE = (128, 128)
IMG_SHAPE = IMG_SIZE + (3,)
base_learning_rate = 0.0001

# datagen = ImageDataGenerator(
#             rotation_range=20,            # Degree range for random rotations
#             width_shift_range=0.2,        # Fraction of total width for random horizontal shifts
#             height_shift_range=0.2,       # Fraction of total height for random vertical shifts
#             shear_range=0.2,              # Shear intensity (shear angle in radians)
#             zoom_range=0.2,               # Random zoom range
#             horizontal_flip=True,         # Randomly flip inputs horizontally
#             fill_mode='nearest',
#             validation_split=0.2
#         )

datagen = ImageDataGenerator()
openml_tensorflow.config.datagen = datagen

openml_tensorflow.config.dir = openml.config.get_cache_directory()+'/datasets/45923/Images/'
# openml_tensorflow.config.dir = 'dataset/Images'
openml_tensorflow.config.x_col = "Filename"
openml_tensorflow.config.y_col = 'Class_encoded'
openml_tensorflow.config.datagen = datagen
openml_tensorflow.config.batch_size = 2
openml_tensorflow.config.class_mode = "categorical"
openml_tensorflow.config.perform_validation = True

kwargs = {
    'callbacks': tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0),
    'verbose': 2
}
openml_tensorflow.config.kwargs = kwargs

IMG_SIZE = 128
NUM_CLASSES = 67

def build_model():
    
    dropout_rate = 0.6

    base = EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        pooling=None)
    count = 0
    count_trainable = 0	
    for layer in base.layers:
        if count >= len(base.layers) - 10:
            layer.trainable = True
            count_trainable += 1
        else:
            layer.trainable = False
        count += 1

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Ensure that you're passing it as a string
              metrics=['accuracy'])
    print(count_trainable)
    return model

############################################################################

# Download the OpenML task for the Indoorscenes dataset.

# task = openml.tasks.get_task(362065)#   10 fold cross validation 
task = openml.tasks.get_task(362070)#   3 fold cross validation

model = build_model()

# Run the model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

# If you want to publish the run with the onnx file, 
# then you must call openml_tensorflow.add_onnx_to_run() immediately before run.publish(). 
# When you publish, onnx file of last trained model is uploaded. 
# Careful to not call this function when another run_model_on_task is called in between, 
# as during publish later, only the last trained model (from last run_model_on_task call) is uploaded.   
run = openml_tensorflow.add_onnx_to_run(run)

run.publish()

print('URL for run: %s/run/%d?api_key=%s' % (openml.config.server, run.run_id, openml.config.apikey))
############################################################################

# Visualize model in netron

from urllib.request import urlretrieve

published_run = openml.runs.get_run(run.run_id)
url = 'https://api.openml.org/data/download/{}/model.onnx'.format(published_run.output_files['onnx_model'])

file_path, _ = urlretrieve(url, 'model.onnx')

import netron
# Visualize the ONNX model using Netron
netron.start(file_path)

# URL for run: https://www.openml.org/api/v1/xml/run/10594206