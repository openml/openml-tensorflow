"""
Keras sequential classification example
==================

An example of a sequential network used as an OpenML flow.
"""

import tensorflow

import openml
import openml_tensorflow
############################################################################
# Define a sequential Keras model.
openml.config.apikey = '033cb8cc8143c53180b10eec84835b2e'
openml_tensorflow.config.epoch = 10

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Reshape((32, 32, 3)),
    tensorflow.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),

    tensorflow.keras.layers.Dropout(0.25),
    tensorflow.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tensorflow.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tensorflow.keras.layers.Dropout(0.25),
    tensorflow.keras.layers.Conv2D(32, (3, 3), padding='same'),
    tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tensorflow.keras.layers.Dropout(0.25),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(units=1024, activation=tensorflow.keras.activations.relu),
    tensorflow.keras.layers.Dropout(rate=0.4),
    tensorflow.keras.layers.Dense(units=10, activation=tensorflow.keras.activations.softmax),
])

# We will compile using the Adam optimizer while targeting accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
############################################################################

############################################################################
# Download the OpenML task for the german credit card dataset.
task = openml.tasks.get_task(167124)
############################################################################
# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
