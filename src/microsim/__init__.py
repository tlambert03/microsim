"""Light microscopy simulation in python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("microsim")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"

from .schema import Simulation

__all__ = ["Simulation", "__version__"]
