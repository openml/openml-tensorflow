import copy
import importlib
import json
import logging
import pickle
import re
import sys
import warnings
import zlib
from collections import OrderedDict  # noqa: F401
from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import tensorflow
import numpy as np
import pandas as pd
import scipy.sparse
from . import config


import openml
from openml.exceptions import PyOpenMLError
from openml.extensions import Extension, register_extension
from openml.flows import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError

DEPENDENCIES_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)((?P<operation>==|>=|>)'
    r'(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$'
)

SIMPLE_NUMPY_TYPES = [nptype for type_cat, nptypes in np.sctypes.items()
                      for nptype in nptypes if type_cat != 'others']
SIMPLE_TYPES = tuple([bool, int, float, str] + SIMPLE_NUMPY_TYPES)

LAYER_PATTERN = re.compile(r'layer\d+\_(.*)')

tensorflow.executing_eagerly()


class TensorflowExtension(Extension):
    """Connect Keras to OpenML-Python."""

    ################################################################################################
    # General setup

    @classmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        """Check whether a given flow describes a Keras neural network.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_tf_flow(flow)

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``tf.models.Model``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, tensorflow.keras.models.Model)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow', initialize_with_defaults: bool = False) -> Any:
        """Initializes a Keras model based on a flow.

        Parameters
        ----------
        flow : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        mixed
        """
        return self._deserialize_tf(flow, initialize_with_defaults=initialize_with_defaults)

    def _deserialize_tf(
            self,
            o: Any,
            components: Optional[Dict] = None,
            initialize_with_defaults: bool = False,
            recursion_depth: int = 0,
    ) -> Any:
        """
        Recursive function to deserialize a tensorflow flow.

        This function delegates all work to the respective functions to deserialize special data
        structures etc.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        components : dict
            empty

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        recursion_depth : int
            The depth at which this flow is called, mostly for debugging
            purposes

        Returns
        -------
        mixed
        """

        logging.info('-%s flow_to_keras START o=%s, components=%s, '
                     'init_defaults=%s' % ('-' * recursion_depth, o, components,
                                           initialize_with_defaults))
        depth_pp = recursion_depth + 1  # shortcut var, depth plus plus

        # First, we need to check whether the presented object is a json string.
        # JSON strings are used to encoder parameter values. By passing around
        # json strings for parameters, we make sure that we can flow_to_keras
        # the parameter values to the correct type.

        if isinstance(o, str):
            try:
                o = json.loads(o)
            except JSONDecodeError:
                pass

        rval = None  # type: Any
        if isinstance(o, dict):
            rval = dict(
                (
                    self._deserialize_tf(
                        o=key,
                        components=components,
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp,
                    ),
                    self._deserialize_tf(
                        o=value,
                        components=components,
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp,
                    )
                )
                for key, value in sorted(o.items())
            )
        elif isinstance(o, (list, tuple)):
            rval = [
                self._deserialize_tf(
                    o=element,
                    components=components,
                    initialize_with_defaults=initialize_with_defaults,
                    recursion_depth=depth_pp,
                )
                for element in o
            ]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, str)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            if not self._is_tf_flow(o):
                raise ValueError('Only Keras flows can be reinstantiated')
            rval = self._deserialize_model(
                flow=o,
                keep_defaults=initialize_with_defaults,
                recursion_depth=recursion_depth,
            )
        else:
            raise TypeError(o)
        logging.info('-%s flow_to_tf END   o=%s, rval=%s'
                     % ('-' * recursion_depth, o, rval))
        return rval

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform a Keras model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types
        return self._serialize_tf(model)

    def _serialize_tf(self, o: Any, parent_model: Optional[Any] = None) -> Any:
        rval = None  # type: Any

        if self.is_estimator(o):
            # is the main model or a submodel
            rval = self._serialize_model(o)
        elif isinstance(o, (list, tuple)):
            rval = [self._serialize_tf(element, parent_model) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, SIMPLE_TYPES) or o is None:
            if isinstance(o, tuple(SIMPLE_NUMPY_TYPES)):
                o = o.item()
            # base parameter values
            rval = o
        elif isinstance(o, dict):
            if not isinstance(o, OrderedDict):
                o = OrderedDict([(key, value) for key, value in sorted(o.items())])

            rval = OrderedDict()
            for key, value in o.items():
                if not isinstance(key, str):
                    raise TypeError('Can only use string as keys, you passed '
                                    'type %s for value %s.' %
                                    (type(key), str(key)))
                key = self._serialize_tf(key, parent_model)
                value = self._serialize_tf(value, parent_model)
                rval[key] = value
            rval = rval
        else:
            raise TypeError(o, type(o))

        return rval

    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Libraries listed are ``Python``, ``tensorflow``, ``numpy`` and ``scipy``.

        Returns
        -------
        List
        """

        import tensorflow
        import scipy
        import numpy

        major, minor, micro, _, _ = sys.version_info
        python_version = 'Python_{}.'.format(
            ".".join([str(major), str(minor), str(micro)]))
        tensorflow_version = 'Keras_{}.'.format(tensorflow.__version__)
        numpy_version = 'NumPy_{}.'.format(numpy.__version__)
        scipy_version = 'SciPy_{}.'.format(scipy.__version__)

        return [python_version, tensorflow_version, numpy_version, scipy_version]

    def create_setup_string(self, model: Any) -> str:
        """Create a string which can be used to reinstantiate the given model.

        Parameters
        ----------
        model : Any

        Returns
        -------
        str
        """
        run_environment = " ".join(self.get_version_information())
        return run_environment + " " + str(model)

    @classmethod
    def _is_tf_flow(cls, flow: OpenMLFlow) -> bool:
        return (flow.external_version.startswith('tensorflow==')
                or ',tensorflow==' in flow.external_version)

    def _serialize_model(self, model: Any) -> OpenMLFlow:
        """Create an OpenMLFlow.

        Calls `tf_to_flow` recursively to properly serialize the
        parameters to strings and the components (other models) to OpenMLFlows.

        Parameters
        ----------
        model : Keras neural network

        Returns
        -------
        OpenMLFlow

        """

        # Get all necessary information about the model objects itself
        parameters, parameters_meta_info, subcomponents, subcomponents_explicit = \
            self._extract_information_from_model(model)

        # Create a flow name, which contains a hash of the parameters as part of the name
        # This is done in order to ensure that we are not exceeding the 1024 character limit
        # of the API, since NNs can become quite large
        class_name = model.__module__ + "." + model.__class__.__name__
        class_name += '.' + format(
            zlib.crc32(json.dumps(parameters, sort_keys=True).encode('utf8')),
            'x'
        )

        external_version = self._get_external_version_string(model, subcomponents)
        name = class_name

        dependencies = '\n'.join([
            self._format_external_version(
                'tensorflow',
                tensorflow.__version__,
            ),
            'numpy>=1.6.1',
            'scipy>=0.9',
        ])

        tensorflow_version = self._format_external_version('tensorflow', tensorflow.__version__)
        tensorflow_version_formatted = tensorflow_version.replace('==', '_')
        flow = OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created tensorflow flow.',
                          model=model,
                          components=subcomponents,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version,
                          tags=['openml-python', 'tensorflow',
                                'python', tensorflow_version_formatted,

                                ],
                          language='English',
                          dependencies=dependencies)

        return flow

    def _get_external_version_string(
            self,
            model: Any,
            sub_components: Dict[str, OpenMLFlow],
    ) -> str:
        # Create external version string for a flow, given the model and the
        # already parsed dictionary of sub_components. Retrieves the external
        # version of all subcomponents, which themselves already contain all
        # requirements for their subcomponents. The external version string is a
        # sorted concatenation of all modules which are present in this run.
        model_package_name = model.__module__.split('.')[0]
        module = importlib.import_module(model_package_name)
        model_package_version_number = module.__version__  # type: ignore
        external_version = self._format_external_version(
            model_package_name, model_package_version_number,
        )
        openml_version = self._format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)

        return ','.join(list(sorted(external_versions)))

    def _from_parameters(self, parameters: 'OrderedDict[str, Any]') -> Any:
        """ Get a tensorflow model from flow parameters """

        # Create a dict and recursively fill it with model components
        # First do this for non-layer items, then layer items.
        config = {}

        # Add the expected configuration parameters back to the configuration dictionary,
        # as long as they are not layers, since they need to be deserialized separately
        for k, v in parameters.items():
            if not LAYER_PATTERN.match(k):
                config[k] = self._deserialize_tf(v)

        # Recreate the layers list and start to deserialize them back to the correct location
        config['config']['layers'] = []
        for k, v in parameters.items():
            if LAYER_PATTERN.match(k):
                v = self._deserialize_tf(v)
                config['config']['layers'].append(v)

        # Deserialize the model from the configuration dictionary
        model = tensorflow.keras.layers.deserialize(config)

        # Attempt to recompile the model if compilation parameters were present
        # during serialization
        if 'optimizer' in parameters:
            training_config = self._deserialize_tf(parameters['optimizer'])
            optimizer_config = training_config['optimizer_config']
            optimizer = tensorflow.keras.optimizers.deserialize(optimizer_config)

            # Recover loss functions and metrics
            loss = training_config['loss']
            metrics = training_config['metrics']
            sample_weight_mode = training_config['sample_weight_mode']
            loss_weights = training_config['loss_weights']

            # Compile model
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode)
        else:
            warnings.warn('No training configuration found inside the flow: '
                          'the model was *not* compiled. '
                          'Compile it manually.')

        return model

    def _get_parameters(self, model: Any) -> 'OrderedDict[str, Optional[str]]':
        # Get the parameters from a model in an OrderedDict
        parameters = OrderedDict()  # type: OrderedDict[str, Any]

        # Construct the configuration dictionary in the same manner as
        # keras.engine.Network.to_json does
        model_config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config(),
            'tensorflow_version': tensorflow.__version__,
            'backend': tensorflow.keras.backend.backend()
        }

        # Remove the layers from the configuration in order to allow them to be
        # pretty printed as model parameters
        layers = model_config['config']['layers']
        del model_config['config']['layers']

        # Add the rest of the model configuration entries to the parameter list
        for k, v in model_config.items():
            parameters[k] = self._serialize_tf(v, model)

        # Compute the format of the layer numbering. This pads the layer numbers with 0s in
        # order to ensure that the layers are printed in a human-friendly order, instead of
        # having weird orderings
        max_len = int(np.ceil(np.log10(len(layers))))
        len_format = '{0:0>' + str(max_len) + '}'

        # Add the layers as hyper-parameters
        for i, v in enumerate(layers):
            layer = v['config']
            k = 'layer' + len_format.format(i) + "_" + layer['name']
            parameters[k] = self._serialize_tf(v, model)

        # Introduce the optimizer settings as hyper-parameters, if the model has been compiled
        if model.optimizer:
            parameters['optimizer'] = self._serialize_tf({
                'optimizer_config': {
                    'class_name': model.optimizer.__class__.__name__,
                    'config': model.optimizer.get_config()
                },
                'loss': model.loss,
                'metrics': model.metrics,
                'weighted_metrics': model.metrics,
                # 'sample_weight_mode': model.sample_weight_mode,
                # 'loss_weights': model.loss_weights,
            }, model)

        return parameters

    def _extract_information_from_model(
            self,
            model: Any,
    ) -> Tuple[
        'OrderedDict[str, Optional[str]]',
        'OrderedDict[str, Optional[Dict]]',
        'OrderedDict[str, OpenMLFlow]',
        Set,
    ]:
        # Stores all entities that should become subcomponents (unused)
        sub_components = OrderedDict()  # type: OrderedDict[str, OpenMLFlow]
        # Stores the keys of all subcomponents that should become (unused)
        sub_components_explicit = set()  # type: Set
        parameters = OrderedDict()  # type: OrderedDict[str, Optional[str]]
        parameters_meta_info = OrderedDict()  # type: OrderedDict[str, Optional[Dict]]

        model_parameters = self._get_parameters(model)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._serialize_tf(v, model)
            rval = json.dumps(rval)

            parameters[k] = rval
            parameters_meta_info[k] = OrderedDict((('description', None), ('data_type', None)))

        return parameters, parameters_meta_info, sub_components, sub_components_explicit

    def _deserialize_model(
            self,
            flow: OpenMLFlow,
            keep_defaults: bool,
            recursion_depth: int,
    ) -> Any:
        logging.info('-%s deserialize %s' % ('-' * recursion_depth, flow.name))
        self._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        components = flow.components
        parameter_dict = OrderedDict()  # type: OrderedDict[str, Any]

        # Do a shallow copy of the components dictionary so we can remove the
        # components from this copy once we added them into the layer list. This
        # allows us to not consider them any more when looping over the
        # components, but keeping the dictionary of components untouched in the
        # original components dictionary.
        components_ = copy.copy(components)

        for name in parameters:
            value = parameters.get(name)
            logging.info('--%s flow_parameter=%s, value=%s' %
                         ('-' * recursion_depth, name, value))
            rval = self._deserialize_tf(
                value,
                components=components_,
                initialize_with_defaults=keep_defaults,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        for name in components:
            if name in parameter_dict:
                continue
            if name not in components_:
                continue
            value = components[name]
            logging.info('--%s flow_component=%s, value=%s'
                         % ('-' * recursion_depth, name, value))
            rval = self._deserialize_tf(
                value,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        return self._from_parameters(parameter_dict)

    def _check_dependencies(self, dependencies: str) -> None:
        """
        Checks whether the dependencies required for the deserialization of an OpenMLFlow are met

        Parameters
        ----------
        dependencies : str
            a string representing the required dependencies

        Returns
        -------
        None
        """
        if not dependencies:
            return

        dependencies_list = dependencies.split('\n')
        for dependency_string in dependencies_list:
            match = DEPENDENCIES_PATTERN.match(dependency_string)
            if not match:
                raise ValueError('Cannot parse dependency %s' % dependency_string)

            dependency_name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')

            module = importlib.import_module(dependency_name)
            required_version = LooseVersion(version)
            installed_version = LooseVersion(module.__version__)  # type: ignore

            if operation == '==':
                check = required_version == installed_version
            elif operation == '>':
                check = installed_version > required_version
            elif operation == '>=':
                check = (installed_version > required_version
                         or installed_version == required_version)
            else:
                raise NotImplementedError(
                    'operation \'%s\' is not supported' % operation)
            if not check:
                raise ValueError('Trying to deserialize a model with dependency '
                                 '%s not satisfied.' % dependency_string)

    def _format_external_version(
            self,
            model_package_name: str,
            model_package_version_number: str,
    ) -> str:
        """
        Returns a formatted string representing the required dependencies for a flow

        Parameters
        ----------
        model_package_name : str
            the name of the required package
        model_package_version_number : str
            the version of the required package
        Returns
        -------
        str
        """
        return '%s==%s' % (model_package_name, model_package_version_number)

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a Keras neural network.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, tensorflow.keras.models.Model)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """
        Not applied for Keras, since there are no random states in Keras.

        Parameters
        ----------
        model : keras model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """

        return model

    def _run_model_on_fold(
            self,
            model: Any,
            task: 'OpenMLTask',
            X_train: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
            rep_no: int,
            fold_no: int,
            y_train: Optional[np.ndarray] = None,
            X_test: Optional[Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        'OrderedDict[str, float]',
        Optional[OpenMLRunTrace],
        Optional[Any]
    ]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Furthermore, it will measure run time measures in case multi-core behaviour allows this.
        * exact user cpu time will be measured if the number of cores is set (recursive throughout
        the model) exactly to 1
        * wall clock time will be measured if the number of cores is set (recursive throughout the
        model) to any given number (but not when it is set to -1)

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are doing.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
        task : OpenMLTask
            The task to run the model on.
        X_train : array-like
            Training data for the given repetition and fold.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        y_train : Optional[np.ndarray] (default=None)
            Target attributes for supervised tasks. In case of classification, these are integer
            indices to the potential classes specified by dataset.
        X_test : Optional, array-like (default=None)
            Test attributes to test for generalization in supervised tasks.

        Returns
        -------
        predictions : np.ndarray
            Model predictions.
        probabilities :  Optional, np.ndarray
            Predicted probabilities (only applicable for supervised classification tasks).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        trace : Optional, OpenMLRunTrace
            Hyperparameter optimization trace (only applicable for supervised tasks with
            hyperparameter optimization).
        additional_information: Optional, Any
            Additional information provided by the extension to be converted into additional files.
        """

        def _prediction_to_probabilities(y: np.ndarray, classes: List[Any]) -> np.ndarray:
            """Transforms predicted probabilities to match with OpenML class indices.

            Parameters
            ----------
            y : np.ndarray
                Predicted probabilities (possibly omitting classes if they were not present in the
                training data).
            model_classes : list
                List of classes known_predicted by the model, ordered by their index.

            Returns
            -------
            np.ndarray
            """
            # y: list or numpy array of predictions
            # model_classes: keras classifier mapping from original array id to
            # prediction index id
            if not isinstance(classes, list):
                raise ValueError('please convert model classes to list prior to '
                                 'calling this fn')
            result = np.zeros((len(y), len(classes)), dtype=np.float32)
            for obs, prediction_idx in enumerate(y):
                result[obs][prediction_idx] = 1.0
            return result

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError('argument y_train must not be of type None')
            if X_test is None:
                raise TypeError('argument X_test must not be of type None')

        # This might look like a hack, and it is, but it maintains the compilation status,
        # in contrast to clone_model, and also is faster than using get_config + load_from_config
        # since it avoids string parsing
        import dill
        import weakref
        model_copy = dill.loads(dill.dumps(model))
        #model_copy = pickle.loads(pickle.dumps(model))

        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'

        try:
            if isinstance(task, OpenMLSupervisedTask):
                epoch = config.epoch
                batch_size = config.batch_size
                model_copy.fit(X_train, y_train, batch_size=config.batch_size,epochs=epoch)
        except AttributeError as e:
            # typically happens when training a regressor on classification task
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLClassificationTask):
            model_classes = tensorflow.keras.backend.argmax(y_train, axis=-1)

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            pred_y = model_copy.predict(X_test)
            if isinstance(task, OpenMLClassificationTask):
                pred_y = tensorflow.keras.backend.argmax(pred_y)
            elif isinstance(task, OpenMLRegressionTask):
                pred_y = tensorflow.keras.backend.reshape(pred_y, (-1,))
            pred_y = tensorflow.keras.backend.eval(pred_y)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):

            try:
                proba_y = model_copy.predict(X_test)
            except AttributeError:
                if task.class_labels is not None:
                    proba_y = _prediction_to_probabilities(pred_y, list(task.class_labels))
                else:
                    raise ValueError('The task has no class labels')
            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing at training time
                    # By default, the classification targets are mapped to be zero-based indices
                    # to the actual classes. Therefore, the model_classes contain the correct
                    # indices to the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities for class 3
                    # (because the rest of the library expects that the probabilities are ordered
                    # the same way as the classes are ordered).
                    proba_y_new = np.zeros((proba_y.shape[0], len(task.class_labels)))
                    for idx, model_class in enumerate(model_classes):
                        proba_y_new[:, model_class] = proba_y[:, idx]
                    proba_y = proba_y_new

                if proba_y.shape[1] != len(task.class_labels):
                    message = "Estimator only predicted for {}/{} classes!".format(
                        proba_y.shape[1], len(task.class_labels),
                    )
                    warnings.warn(message)
                    openml.config.logger.warn(message)

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None
        else:
            raise TypeError(type(task))

        return pred_y, proba_y, user_defined_measures, None

    def compile_additional_information(
            self,
            task: 'OpenMLTask',
            additional_information: List[Tuple[int, int, Any]]
    ) -> Dict[str, Tuple[str, str]]:
        """Compiles additional information provided by the extension during the runs into a final
        set of files.

        Parameters
        ----------
        task : OpenMLTask
            The task the model was run on.
        additional_information: List[Tuple[int, int, Any]]
            A list of (fold, repetition, additional information) tuples obtained during training.

        Returns
        -------
        files : Dict[str, Tuple[str, str]]
            A dictionary of files with their file name and contents.
        """
        return dict()

    def obtain_parameter_values(
            self,
            flow: 'OpenMLFlow',
            model: Any = None,
    ) -> List[Dict[str, Any]]:
        """Extracts all parameter settings required for the flow from the model.

        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.

        Parameters
        ----------
        flow : OpenMLFlow
            OpenMLFlow object (containing flow ids, i.e., it has to be downloaded from the server)

        model: Any, optional (default=None)
            The model from which to obtain the parameter values. Must match the flow signature.
            If None, use the model specified in ``OpenMLFlow.model``.

        Returns
        -------
        list
            A list of dicts, where each dict has the following entries:
            - ``oml:name`` : str: The OpenML parameter name
            - ``oml:value`` : mixed: A representation of the parameter value
            - ``oml:component`` : int: flow id to which the parameter belongs
        """
        openml.flows.functions._check_flow_for_server_id(flow)

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(_flow, _flow_dict, component_model,
                               _main_call=False, main_id=None):
            # _flow is openml flow object, _param dict maps from flow name to flow
            # id for the main call, the param dict can be overridden (useful for
            # unit tests / sentinels) this way, for flows without subflows we do
            # not have to rely on _flow_dict
            exp_parameters = set(_flow.parameters)
            exp_components = set(_flow.components)

            _model_parameters = self._get_parameters(component_model)

            model_parameters = set(_model_parameters.keys())
            if len((exp_parameters | exp_components) ^ model_parameters) != 0:
                flow_params = sorted(exp_parameters | exp_components)
                model_params = sorted(model_parameters)
                raise ValueError('Parameters of the model do not match the '
                                 'parameters expected by the '
                                 'flow:\nexpected flow parameters: '
                                 '%s\nmodel parameters: %s' % (flow_params,
                                                               model_params))

            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current['oml:name'] = _param_name

                current_param_values = self.model_to_flow(_model_parameters[_param_name])

                # Try to filter out components (a.k.a. subflows) which are
                # handled further down in the code (by recursively calling
                # this function)!
                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                # vanilla parameter value
                parsed_values = json.dumps(current_param_values)

                _current['oml:value'] = parsed_values
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = self._get_parameters(component_model)[_identifier]
                _params.extend(extract_parameters(_flow.components[_identifier],
                                                  _flow_dict, subcomponent_model))
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        parameters = extract_parameters(flow, flow_dict, model, True, flow.flow_id)

        return parameters

    def _openml_param_name_to_keras(
            self,
            openml_parameter: openml.setups.OpenMLParameter,
            flow: OpenMLFlow,
    ) -> str:
        """
        Converts the name of an OpenMLParameter into the Keras name, given a flow.

        Parameters
        ----------
        openml_parameter: OpenMLParameter
            The parameter under consideration

        flow: OpenMLFlow
            The flow that provides context.

        Returns
        -------
        keras_parameter_name: str
            The name the parameter will have once used in Keras
        """
        if not isinstance(openml_parameter, openml.setups.OpenMLParameter):
            raise ValueError('openml_parameter should be an instance of OpenMLParameter')
        if not isinstance(flow, OpenMLFlow):
            raise ValueError('flow should be an instance of OpenMLFlow')

        flow_structure = flow.get_structure('name')
        if openml_parameter.flow_name not in flow_structure:
            raise ValueError('Obtained OpenMLParameter and OpenMLFlow do not correspond. ')
        name = openml_parameter.flow_name  # for PEP8
        return '__'.join(flow_structure[name] + [openml_parameter.parameter_name])

    def instantiate_model_from_hpo_class(
            self,
            model: Any,
            trace_iteration: OpenMLTraceIteration,
    ) -> Any:
        """Instantiate a ``base_estimator`` which can be searched over by the hyperparameter
        optimization model (UNUSED)

        Parameters
        ----------
        model : Any
            A hyperparameter optimization model which defines the model to be instantiated.
        trace_iteration : OpenMLTraceIteration
            Describing the hyperparameter settings to instantiate.

        Returns
        -------
        Any
        """

        return model
    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained
        Parameters
        ----------
        model : Any
        Returns
        -------
        bool
        """

