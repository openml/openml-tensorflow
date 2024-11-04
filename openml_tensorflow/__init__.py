
import os
from .extension import TensorflowExtension
from openml.extensions import register_extension
from . import config, extension

__all__ = ['TensorflowExtension', 'config','add_onnx_to_run']

register_extension(TensorflowExtension)

def add_onnx_to_run(run):
    
    run._old_get_file_elements = run._get_file_elements
    
    def modified_get_file_elements():
        onnx_ = extension.last_models # saving as local variable to solve RecursionError: maximum recursion depth exceeded
        elements = run._old_get_file_elements()
        elements["onnx_model"] = ("model.onnx", onnx_)
        return elements
    
    run._get_file_elements = modified_get_file_elements
    return run
    