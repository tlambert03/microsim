from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import (
        GetCoreSchemaHandler,
        GetJsonSchemaHandler,
    )
    from pydantic.json_schema import JsonSchemaValue


# TODO: add generic typing for dtype, shape, pint.Unit, etc...
class _NumpyNdarrayPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate_from_any(value: Any) -> np.ndarray:
            try:
                return np.asarray(value)
            except Exception as e:
                raise ValueError(f"Cannot cast {value} to numpy.ndarray: {e}") from e

        from_any_schema = core_schema.chain_schema(
            [
                core_schema.any_schema(),
                core_schema.no_info_plain_validator_function(validate_from_any),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_any_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(np.ndarray),
                    from_any_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.tolist()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for arrays
        return handler(core_schema.list_schema(core_schema.any_schema()))


# We now create an `Annotated` wrapper that we'll use as the annotation for fields on
# `BaseModel`s, etc.
NumpyNdarray = Annotated[np.ndarray, _NumpyNdarrayPydanticAnnotation]
