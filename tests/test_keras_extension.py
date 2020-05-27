import sys
import os
import openml
from openml_keras import KerasExtension
from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow
from openml.flows.functions import assert_flows_equal
from openml.runs.trace import OpenMLRunTrace
from openml.testing import TestBase, SimpleImputer
import keras
from collections import OrderedDict
import unittest


this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)

__version__ = 0.1


class TestKerasExtensionFlowFunctions(unittest.TestCase):
    # def setUp(self):
    #     super().setUp(n_levels=2)
    #     self.extension = KerasExtension()

    def test_serialize_model(self):
        self.extension = KerasExtension()

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
        fixture_name = 'keras.engine.sequential.Sequential.3004bddd'
        fixture_description = 'Automatically created keras flow.'
        version_fixture = 'keras==2.2.4' \
                          'numpy>=1.6.1' \
                          'scipy>=0.9 '
        fixture_parameters = OrderedDict((('backend','"tensorflow"'),
                                          ('class_name','"Sequential"'),
                                          ('config','{"name": "sequential_1"}'),
                                          ('keras_version','"2.2.4"'),
                                          ('layer0_batch_normalization_1','{"class_name": "BatchNormalization", "config": {"axis": -1, "beta_constraint": null, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_regularizer": null, "center": true, "epsilon": 0.001, "gamma_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_regularizer": null, "momentum": 0.99, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "name": "batch_normalization_1", "scale": true, "trainable": true}}'),
                                          ('layer1_dense_1','{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "kernel_constraint": null, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "mode": "fan_avg", "scale": 1.0, "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 1024, "use_bias": true}}'),
                                          ('layer2_dropout_1','{"class_name": "Dropout", "config": {"name": "dropout_1", "noise_shape": null, "rate": 0.4, "seed": null, "trainable": true}}'),
                                          ('layer3_dense_2','{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "kernel_constraint": null, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "mode": "fan_avg", "scale": 1.0, "seed": null}}, "kernel_regularizer": null, "name": "dense_2", "trainable": true, "units": 2, "use_bias": true}}'),
                                          ('optimizer','{"loss": "sparse_categorical_crossentropy", "loss_weights": null, "metrics": ["accuracy"], "optimizer_config": {"class_name": "Adam", "config": {"amsgrad": false, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "decay": 0.0, "epsilon": 1e-07, "lr": 0.0010000000474974513}}, "sample_weight_mode": null, "weighted_metrics": null}')
                                          ))
        structure_fixture = {'keras.engine.sequential.Sequential.3004bddd':[]}
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

