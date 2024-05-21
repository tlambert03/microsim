"""Functions that return coerced/validated values."""

from typing import Any

import numpy as np


def validate_np_dtype(x: Any) -> np.dtype:
    try:
        return np.dtype(x)
    except Exception as e:
        raise ValueError(f"Cannot cast {x!r} to a numpy dtype: {e}") from e


def validate_np_floating_dtype(x: Any) -> np.dtype[np.floating]:
    dt = validate_np_dtype(x)
    if not np.issubdtype(dt, np.floating):
        raise ValueError(f"Expected a floating-point dtype, got {dt}")
    return dt


def validate_np_integer_dtype(x: Any) -> np.dtype[np.integer]:
    dt = validate_np_dtype(x)
    if not np.issubdtype(dt, np.floating):
        raise ValueError(f"Expected a integer dtype, got {dt}")
    return dt
