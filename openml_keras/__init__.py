
import os
from .extension import KerasExtension
from openml.extensions import register_extension

__all__ = ['KerasExtension']

register_extension(KerasExtension)
