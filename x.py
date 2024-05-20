from typing import Annotated, Any, Generic, TypeVar

import numpy as np
import pint
from pint._typing import UnitLike
from pint.facets.plain import MagnitudeT, PlainQuantity
from pydantic import BaseModel, TypeAdapter
from pydantic.annotated_handlers import GetCoreSchemaHandler
from pydantic_core import core_schema

T = TypeVar("T", bound=pint.Unit)


class Unit(Generic[MagnitudeT]):
    def __init__(self, unit: UnitLike, mag_type: MagnitudeT | Any = Any) -> None:
        self.unit = unit
        self.mag_type = mag_type

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _cast_quantity(value: Any) -> PlainQuantity[MagnitudeT]:
            return pint.Quantity(value, self.unit)

        schema: core_schema.CoreSchema
        if isinstance(self.mag_type, type) and issubclass(self.mag_type, np.ndarray):
            schema = core_schema.no_info_before_validator_function(
                np.asarray,
                schema=core_schema.list_schema(core_schema.float_schema()),
            )
        else:
            schema = TypeAdapter(self.mag_type).core_schema

        return core_schema.no_info_after_validator_function(
            _cast_quantity, schema=schema
        )


Meter = Annotated[pint.Quantity, Unit("m")]


class Model(BaseModel):
    meters: Meter
