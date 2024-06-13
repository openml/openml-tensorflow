"""
Tensorflow image classification using pre-trained model example
==================

An example of a tensorflow network that classifies indoor scenes images.
"""

import openml
import openml_tensorflow
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import optimizers, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

############################################################################
openml.config.apikey = 'KEY' # Paste your API key here

############################################################################

openml_tensorflow.config.epoch = 1 #  small epoch for test runs

datagen = ImageDataGenerator()
openml_tensorflow.config.datagen = datagen
openml_tensorflow.config.dir = openml.config.get_cache_directory()+'/datasets/45923/Images/'
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
base_learning_rate = 0.0001

# Example pre-trained model   
base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        weights="imagenet",
                        include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizers.Adam(learning_rate=4e-4),
                loss='categorical_crossentropy',
                metrics=['AUC'])

############################################################################

# Download the OpenML task for the Indoor Scenes dataset.
task = openml.tasks.get_task(362070)#   3 fold cross validation

model = model

# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

"""
Note: If you want to publish the run with the onnx file, 
then you must call openml_tensorflow.add_onnx_to_run() immediately before run.publish(). 
When you publish, onnx file of last trained model is uploaded. 
Careful to not call this function when another run_model_on_task is called in between, 
as during publish later, only the last trained model (from last run_model_on_task call) is uploaded.   
"""
run = openml_tensorflow.add_onnx_to_run(run)

run.publish()

print('URL for run: %s/run/%d?api_key=%s' % (openml.config.server, run.run_id, openml.config.apikey))

############################################################################
# Optional: Visualize model in netron

from urllib.request import urlretrieve

published_run = openml.runs.get_run(run.run_id)
url = 'https://api.openml.org/data/download/{}/model.onnx'.format(published_run.output_files['onnx_model'])

file_path, _ = urlretrieve(url, 'model.onnx')

import netron
# Visualize the ONNX model using Netron
netron.start(file_path)
