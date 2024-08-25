from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from types import FunctionType, ModuleType
from typing import Any


class FFTBackend(ABC):
    """Base class for FFT backends.

    Implements https://uarray.org/en/latest/libauthor_docs.html
    """

    __ua_domain__ = "numpy.scipy.fft"

    @abstractmethod
    def fft_module(self) -> ModuleType:
        raise NotImplementedError

    @abstractmethod
    def convert(self, value: Any, type: Any) -> Any:
        raise NotImplementedError

    def __ua_function__(self, method: FunctionType, args: Any, kwargs: Any) -> Any:
        """Dispatch to the appropriate function in the backend module.

        This defines the implementation of a multimethod. `method` is the multimethod
        being called, and it is guaranteed that it is in the same domain as the backend.
        `args` and `kwargs` are the arguments to the function, possibly after conversion
        (explained below)

        Returning NotImplemented signals that the backend does not support this
        operation.
        """
        if fn := getattr(self.fft_module(), method.__name__, None):
            return self.execute(fn, *args, **kwargs)
        return NotImplemented

    def __ua_convert__(self, dispatchables: Iterable, coerce: bool) -> Any:
        """Convert dispatchables to the backend's array type.

        https://uarray.org/en/latest/libauthor_docs.html

        All dispatchable arguments are passed through `__ua_convert__` before being
        passed into `__ua_function__`. `dispatchables` is iterable of Dispatchable and
        `coerce` is whether or not to coerce forcefully. By convention, operations
        larger than O(log n) (where n is the size of the object in memory) should only
        be done if coerce is True. In addition, there are arguments wrapped as
        non-coercible via the coercible attribute, if these must be coerced, then one
        should return NotImplemented.

        Returning NotImplemented signals that the backend does not support the
        conversion of the given object.

        """
        if coerce:
            return [self.convert(d.value, d.type) for d in dispatchables]
        return NotImplemented

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class JaxFFTBackend(FFTBackend):
    def __init__(self, device: Any = None) -> None:
        try:
            import jax
        except ImportError as e:
            raise ImportError("JaxFFTBackend requires JAX to be installed") from e
        self.jax = jax
        self.device = jax.devices(device)[0] if device else None

    def fft_module(self) -> ModuleType:
        return self.jax.numpy.fft  # type: ignore [no-any-return]

    def convert(self, value: Any, type: type) -> Any:
        result = self.jax.numpy.asarray(value)
        if self.device:
            result = self.jax.device_put(result, self.device)
        return result


class TorchFFTBackend(FFTBackend):
    def __init__(self, device: Any = None) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("TorchFFTBackend requires torch to be installed") from e
        self.torch = torch
        self.device = device

    def fft_module(self) -> ModuleType:
        return self.torch.fft  # type: ignore [no-any-return]

    def convert(self, value: Any, type: type) -> Any:
        if self.device == "mps" and value.dtype.itemsize > 4:
            dtype = self.torch.float32
        else:
            dtype = None
        return self.torch.as_tensor(value, device=self.device, dtype=dtype)

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if "axes" in kwargs:
            kwargs["dim"] = kwargs.pop("axes")
        return func(*args, **kwargs)
