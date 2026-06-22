"""Light microscopy simulation in python."""

from importlib.metadata import PackageNotFoundError, version

# microsim is JAX-only and aims for physical accuracy, so enable 64-bit
# precision globally (must happen before any jax array is created).
import jax as _jax

_jax.config.update("jax_enable_x64", True)

try:
    __version__ = version("microsim")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"

from .schema import Simulation  # noqa: E402

__all__ = ["Simulation", "__version__"]
