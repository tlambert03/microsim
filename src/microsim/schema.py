from typing import Annotated, Any, Literal, Mapping, cast

import numpy as np
import numpy.typing as npt
import xarray
from annotated_types import Ge
from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    ValidatorFunctionWrapHandler,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from microsim.samples import MatsLines


Sample = MatsLines






