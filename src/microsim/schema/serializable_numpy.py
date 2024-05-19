"""Serializable numpy array with pydantic.

A temporary workaround to support serializing numpy arrays with pydantic.
Taken from https://github.com/pydantic/pydantic/issues/7017.
"""

from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator, PlainSerializer


def nd_array_custom_before_validator(x: Any) -> Any:
    # custome before validation logic
    return x


def nd_array_custom_serializer(x: np.ndarray) -> str:
    # custome serialization logic
    return str(x)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
