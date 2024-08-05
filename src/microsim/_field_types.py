from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, functional_validators
from pydantic_core import core_schema

from .schema import _validators

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue


class ProvidesCoreSchema(Protocol):
    func: core_schema.NoInfoValidatorFunction | core_schema.WithInfoValidatorFunction

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema: ...


class _ValidatorMixin:
    """Before/AfterValidator mixin for pydantic validators.

    This mixin provides some things I wish that that the `*Validator` classes in
    pydantic.functional_validators provided.  Specifically, it's currently difficult
    to use a simple Annotated[ThirdPartyType, BeforeValidator(...)] with a
    def validator(x: Any) -> Any: ...  because __get_pydantic_core_schema__ will raise
    an exception when gathering schema via handler(source_type) because ThirdPartyType
    is not a recognized type.
    This mixin allows the source_type to be overridden
    1. manually, by setting the `source_type` attribute on the class
    2. by inferring the source_type from the first parameter of the function

    It also defaults the serialization schema to to_string_ser_schema()
    but that's just an internal convenience... you could also use stuff from
    functional_serializers
    """

    source_type: type | None = None
    serialization: core_schema.SerSchema = core_schema.to_string_ser_schema()

    def _get_mode(self) -> Literal["plain", "before", "after"]:
        for base in type(self).__mro__:
            if base.__name__.startswith("AfterVal"):
                return "after"
            if base.__name__.startswith("BeforeVal"):
                return "before"
        return "plain"

    def __get_pydantic_core_schema__(
        self: ProvidesCoreSchema, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        mode = self._get_mode()  # type: ignore[attr-defined]

        # if this is a BeforeValidator, then the source_type is the type of the first
        # parameter of the function (or Any if that fails)
        if src_type := getattr(self, "source_type", None):
            # override the source_type if it was provided
            source_type = src_type
        elif mode != "after":
            # in the case of an AfterValidator, the source_type is the
            # origin type in the Annotated type
            # but for a before validator, we need to infer the source_type
            # from the validator function itself, or default to Any
            try:
                p0 = next(iter(signature(self.func).parameters.values()))
                source_type = p0.annotation
            except Exception:
                source_type = Any

        schema = super().__get_pydantic_core_schema__(source_type, handler)  # type: ignore[safe-super]
        if ser_schema := getattr(self, "serialization", None):
            schema = {**schema, "serialization": ser_schema}
        return schema


class BeforeValidator(_ValidatorMixin, functional_validators.BeforeValidator):
    pass


class AfterValidator(_ValidatorMixin, functional_validators.AfterValidator):
    pass


# TODO: add generic typing for dtype, shape, pint.Unit, etc...
# this is a very simple implementation.
# See also: https://github.com/caniko/pydantic-numpy
# but I'm not sure I want to depend on that or use it directly
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


######################### Custom Field Types #########################

############### numpy stuff ###############

Dtype = Annotated[np.dtype, BeforeValidator(_validators.validate_np_dtype)]
FloatDtype = Annotated[
    np.dtype[np.floating], BeforeValidator(_validators.validate_np_floating_dtype)
]
IntDtype = Annotated[
    np.dtype[np.integer], BeforeValidator(_validators.validate_np_integer_dtype)
]

NumpyNdarray = Annotated[np.ndarray, _NumpyNdarrayPydanticAnnotation]

############### pint stuff ###############

# these are all ultimately also numeric and/or array[numeric] types too

# Meters = Annotated[PlainQuantity, BeforeValidator(_validators.validate_meters)]
# Microns = Annotated[PlainQuantity, BeforeValidator(_validators.validate_microns)]
# Nanometers = Annotated[PlainQuantity, BeforeValidator(_validators.validate_nm)]
# ExtCoeff = Annotated[PlainQuantity, BeforeValidator(_validators.validate_ext_coeff)]
# Nanoseconds = Annotated[PlainQuantity, BeforeValidator(_validators.validate_ns)]
# Seconds = Annotated[PlainQuantity, BeforeValidator(_validators.validate_seconds)]

# Watts = Annotated[PlainQuantity, BeforeValidator(_validators.validate_watts)]
# Watts_cm2 = Annotated[PlainQuantity, BeforeValidator(_validators.validate_irradiance)]
