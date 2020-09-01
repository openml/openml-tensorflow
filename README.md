# Tensorflow extension for OpenML python

Tensorflow extension for [openml-python API](https://github.com/openml/openml-python).

#### Installation Instructions:

`pip install openml-tensorflow`

PyPi link https://pypi.org/project/openml-keras/

#### Usage
Import openML libraries
```python
import openml
import openml_tensorflow
```
Create  and compile a keras model
```python
model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dense(units=1024, activation=tensorflow.keras.activations.relu),
    tensorflow.keras.layers.Dropout(rate=0.4),
    tensorflow.keras.layers.Dense(units=2, activation=tensorflow.keras.activations.softmax),
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

This library is currently under development, please report any bugs or feature reuest in issues section.