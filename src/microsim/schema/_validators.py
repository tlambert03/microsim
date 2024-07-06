"""Functions that return coerced/validated values."""

from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

import numpy as np
import pint
from pint._typing import UnitLike
from pint.facets.plain.quantity import MagnitudeT, PlainQuantity

T = TypeVar("T", bound=pint.Unit)


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


def make_unit_validator(units: UnitLike) -> Callable[[Any], PlainQuantity]:
    """Return a function that casts a value to a pint.Quantity with the given units."""

    def validate_unit(value: Any) -> PlainQuantity[MagnitudeT]:
        """Cast a `value` to a pint.Quantity with the given units."""
        quant: PlainQuantity = pint.Quantity(value)
        if quant.dimensionless:
            quant = pint.Quantity(value, units=units)
        if not quant.check(units):
            raise ValueError(f"Expected a quantity with units {units}, got {quant}")
        with suppress(pint.UndefinedUnitError):
            # try to cast to the given units.
            # This may fail, even if quant.check() passed, if `units` is just a dimenson
            quant = quant.to(units)
        return quant

    return validate_unit


validate_meters = make_unit_validator("meter")
validate_microns = make_unit_validator("um")
validate_nm = make_unit_validator("nm")
validate_ext_coeff = make_unit_validator("1 / M / cm")

validate_ns = make_unit_validator("ns")
validate_seconds = make_unit_validator("second")
validate_watts = make_unit_validator("watt")
validate_irradiance = make_unit_validator("W/cm^2")
