"""
Keras sequential classification example
==================

An example of a sequential network used as an OpenML flow.
"""

import keras

import openml
import openml_keras
############################################################################
# Define a sequential Keras model.
model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=1024, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=2, activation=keras.activations.softmax),
])

# We will compile using the Adam optimizer while targeting accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
############################################################################

############################################################################
# Download the OpenML task for the german credit card dataset.
task = openml.tasks.get_task(31)
############################################################################
# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
