from typing import Any, no_type_check
from unittest.mock import patch

import numpy.typing as npt
from scipy.signal import fftconvolve
from scipy.signal._signaltools import _centered


@no_type_check
def _apply_conv_mode_no_copy(ret, s1, s2, mode, axes):
    # same as scipy.signal._signaltools._apply_conv_mode but without the .copy() call
    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid)
    raise ValueError("acceptable mode flags are 'valid'," " 'same', or 'full'")


def patched_fftconvolve(
    in1: Any, in2: Any, mode: str = "full", axes: int | npt.ArrayLike | None = None
) -> Any:
    # scipy.signal.fftconvolve assumes that all arraylike objects have a .copy() method
    # which is not the case for some backends (like Torch).
    # This function patches the _apply_conv_mode function to avoid calling .copy()
    # after cropping the result to the desired shape.
    with patch("scipy.signal._signaltools._apply_conv_mode", _apply_conv_mode_no_copy):
        return fftconvolve(in1, in2, mode=mode, axes=axes)
