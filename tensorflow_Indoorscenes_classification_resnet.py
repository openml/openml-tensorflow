"""
Tensorflow image classification model example
==================

An example of a tensorflow network that classifies indoor scenes images.
"""


import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow

import openml
import openml_tensorflow

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from keras import regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers, optimizers, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
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

base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        weights="imagenet",
                        include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
# New model for fine-tuning
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizers.Adam(learning_rate=4e-4),
                loss='categorical_crossentropy',
                metrics=['AUC'])

def build_model_2a():
    # Define the base pre-trained model
    IMG_SIZE_2 = 128
    base_model = EfficientNetV2B2(input_shape=(IMG_SIZE_2, IMG_SIZE_2, 3),
                                  include_top=False, weights="imagenet")
    # Add layers on top of the base model
    input_tensor = base_model.input
    # input_tensor = layers.Input(shape=(IMG_SIZE_2, IMG_SIZE_2, 3))
    x = base_model.output
    # Add more layers here as needed
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Define the final model
    model = models.Model(inputs=input_tensor, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

 
def build_model_2b():
    IMG_SIZE_2 = 128
    base_model = EfficientNetV2B2(input_shape=(IMG_SIZE_2, IMG_SIZE_2, 3),
                                  include_top=False, weights="imagenet")
    
    # input_tensor = layers.Input(shape=(IMG_SIZE_2, IMG_SIZE_2, 3))
    input_tensor = base_model.input

    x = base_model(input_tensor, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Define the final model
    model = models.Model(inputs=input_tensor, outputs=x)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model_2c():
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import EfficientNetB2

    IMG_SIZE = 128
    NUM_CLASSES = 67

    base_model = EfficientNetB2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    
    # Freeze convolutional layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # inputs = base_model.input
    base_output = base_model(inputs)
    
    x = GlobalAveragePooling2D()(base_output)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model  

def build_model_2d():
    
    from tensorflow.keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from keras.applications.densenet import DenseNet121
    from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout
    from keras import Input
    from keras.models import Model
    from keras.optimizers.legacy import RMSprop
    from keras.applications.densenet import preprocess_input
    # Input layer
    input_layer = Input(shape=(128, 128, 3))
    preprocess_layer = Lambda(preprocess_input)(input_layer)
    
    # Load the DenseNet121 model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3), input_tensor=preprocess_layer)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Regularize with dropout
    outputs = Dense(67, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_2e():
    net = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        weights="imagenet",
                        include_top=False)

    model = models.Sequential()
    model.add(net)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=4e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    return model

def build_model_23():
    
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

# Download the OpenML task for the Meta_Album_PNU_Micro dataset.

# task = openml.tasks.get_task(362065)#   10 fold cross validation 
task = openml.tasks.get_task(362070)#   3 fold cross validation

model = build_model_23()

# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

# If you want to publish the run with the onnx file, 
# then you must call openml_tensorflow.add_onnx_to_run() immediately before run.publish(). 
# When you publish, onnx file of last trained model is uploaded. 
# Careful to not call this function when another run_model_on_task is called in between, 
# as during publish later, only the last trained model (from last run_model_on_task call) is uploaded.   
run = openml_tensorflow.add_onnx_to_run(run)

run.publish()

print('URL for run: %s/run/%d?api_key=%s' % (openml.config.server, run.run_id, openml.config.apikey))
breakpoint()
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