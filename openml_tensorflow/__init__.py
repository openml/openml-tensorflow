
import os
from .extension import TensorflowExtension
from openml.extensions import register_extension
from . import config, extension
import tf2onnx
__all__ = ['TensorflowExtension', 'config','add_onnx_to_run']

register_extension(TensorflowExtension)

def add_onnx_to_run(run):
    # Convert the TensorFlow model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(extension.last_models, opset=13)
    onnx = onnx_model.SerializeToString()

    run._old_get_file_elements = run._get_file_elements
    
    def modified_get_file_elements():
        elements = run._old_get_file_elements()
        elements["onnx_model"] = ("model.onnx", onnx)
        return elements
    
    run._get_file_elements = modified_get_file_elements
    return run
    