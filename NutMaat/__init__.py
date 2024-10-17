# Import key modules
from .spectrum import *
from .classifier import *
from ._version import __version__

# Define the public API
__all__ = ['spectrum', 'classifier', '_version']