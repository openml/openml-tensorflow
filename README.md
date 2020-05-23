# Keras extension for OpenML python

Keras extension for openml-python API.

#### Installation Instructions:

`pip install openml-keras`

PyPi link https://pypi.org/project/openml-keras/

#### Usage
Import openML libraries
```python
import openml
import openml_keras
```
Create  and compile a keras model
```python
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
```
Download the task from openML and run the model on task.
```python
task = openml.tasks.get_task(3573)
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()
print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
```
