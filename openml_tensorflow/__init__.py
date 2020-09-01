
import os
from .extension import TFExtension
from openml.extensions import register_extension

__all__ = ['TFExtension']

register_extension(TFExtension)
