import openml_tensorflow

import openml
import tensorflow.keras as keras
import tensorflow as tf

# openml_keras.config.epoch = 10
# openml_keras.config.batch_size = 64
model = keras.models.Sequential([
    keras.layers.Reshape((96,96,3), input_shape=(27648, )),
    keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.Conv2D(64,(3,3), padding = 'same'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(32,(3,3), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),

    # keras.layers.Conv2D(32, (3,3), padding='same'),
    keras.layers.Dense(units=2, activation=keras.activations.softmax),
])

# We will compile using the Adam optimizer while targeting accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              )
############################################################################
openml.config.apikey = '033cb8cc8143c53180b10eec84835b2e'
############################################################################
task = openml.tasks.get_task(360110)
############################################################################
# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))