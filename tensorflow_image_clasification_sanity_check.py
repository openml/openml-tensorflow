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
model.add(layers.Dense(67, activation='softmax'))  # Adjust output size
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# task = openml.tasks.get_task(362071)

from sklearn.model_selection import train_test_split

openml_dataset = openml.datasets.get_dataset(45923, download_all_files=True)
df, *_ = openml_dataset.get_data()

# Local directory with the images
data_dir = os.path.join(os.path.dirname(openml_dataset.data_file), "Images")

# Splitting the data
df_train, df_valid = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Class_name'])

datagen_train = ImageDataGenerator() # You can add data augmentation options here.
train_generator = datagen_train.flow_from_dataframe(dataframe=df_train,
                                            directory=data_dir,
                                            x_col="Filename", y_col="Class_encoded",
                                            class_mode="categorical",
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=32)

history = model.fit(train_generator, steps_per_epoch=openml_tensorflow.config.step_per_epoch,
                                batch_size=openml_tensorflow.config.batch_size, epochs=openml_tensorflow.config.epoch, verbose=1)
learning_curves = history.history