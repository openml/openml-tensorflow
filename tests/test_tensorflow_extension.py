import sys
import os
import openml
from openml_tensorflow import TensorflowExtension
from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow
from openml.flows.functions import assert_flows_equal
from openml.runs.trace import OpenMLRunTrace
from openml.testing import TestBase, SimpleImputer
import tensorflow
from collections import OrderedDict
import unittest

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)

__version__ = 1.0


class TestTfExtensionFlowFunctions(unittest.TestCase):

    def test_serialize_model(self):
        self.extension = TensorflowExtension()

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
     
        fixture_name = 'tensorflow.keras.src.engine.sequential.Sequential.77370b05'
        fixture_description = 'Automatically created tensorflow flow.'
        version_fixture = 'tensorflow==2.15.0' \
                          'numpy>=1.6.1' \
                          'scipy>=0.9'
        fixture_parameters = OrderedDict((('backend','"tensorflow"'),
                                          ('class_name','"Sequential"'),
                                          ('config','{"name": "sequential"}'),
                                          ('layer0_batch_normalization','{"class_name": "BatchNormalization", "config": {"axis": -1, "beta_constraint": null, "beta_initializer": {"class_name": "Zeros", "config": {}, "module": "keras.initializers", "registered_name": null}, "beta_regularizer": null, "center": true, "dtype": "float32", "epsilon": 0.001, "gamma_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {}, "module": "keras.initializers", "registered_name": null}, "gamma_regularizer": null, "momentum": 0.99, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "module": "keras.initializers", "registered_name": null}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "module": "keras.initializers", "registered_name": null}, "name": "batch_normalization", "scale": true, "trainable": true}, "module": "keras.layers", "registered_name": null}'),
                                          ('layer1_dense','{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "module": "keras.initializers", "registered_name": null}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "module": "keras.initializers", "registered_name": null}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 1024, "use_bias": true}, "module": "keras.layers", "registered_name": null}'),
                                          ('layer2_dropout','{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout", "noise_shape": null, "rate": 0.4, "seed": null, "trainable": true}, "module": "keras.layers", "registered_name": null}'),
                                          ('layer3_dense_1','{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "module": "keras.initializers", "registered_name": null}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "module": "keras.initializers", "registered_name": null}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 2, "use_bias": true}, "module": "keras.layers", "registered_name": null}'),
                                          ('optimizer','{"loss": "sparse_categorical_crossentropy", "metrics": [], "optimizer_config": {"class_name": "Adam", "config": {"amsgrad": false, "beta_1": 0.9, "beta_2": 0.999, "clipnorm": null, "clipvalue": null, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "epsilon": 1e-07, "global_clipnorm": null, "is_legacy_optimizer": false, "jit_compile": true, "learning_rate": 0.0010000000474974513, "name": "Adam", "use_ema": false, "weight_decay": null}}}'),
                                          ('tensorflow_version', '"2.15.0"')
                                          ))
        structure_fixture = {'tensorflow.keras.src.engine.sequential.Sequential.77370b05':[]}
        serialization = self.extension.model_to_flow(model)
        structure = serialization.get_structure('name')
        print(serialization.dependencies)
        
        assert serialization.name == fixture_name
        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.class_name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters, fixture_parameters)
        # self.assertEqual(serialization.dependencies, version_fixture)
        self.assertDictEqual(structure, structure_fixture)

        new_model = self.extension.flow_to_model(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

