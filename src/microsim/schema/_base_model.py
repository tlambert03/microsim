import warnings
from pathlib import Path
from typing import Any, ClassVar

import pydantic
from pydantic import BaseModel, ConfigDict, ModelWrapValidatorHandler, model_validator
from pydantic_core import to_jsonable_python

# point warnings at user code, not at microsim/pydantic internals
_SKIP_PREFIXES = (str(Path(__file__).parents[1]), str(Path(pydantic.__file__).parent))


class SimBaseModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",  # never silently ignore unknown keys
    )

    # deprecated field names: {old_name: new_name}
    _renamed_fields: ClassVar[dict[str, str]] = {}

    @model_validator(mode="wrap")
    @classmethod
    def _validate_input_keys(cls, data: Any, handler: ModelWrapValidatorHandler) -> Any:
        """Convert renamed fields, and allow computed fields if they match."""
        given: dict[str, Any] = {}
        if isinstance(data, dict):
            data = cls._convert_renamed_fields(data)
            if computed := cls.model_computed_fields:
                given = {k: v for k, v in data.items() if k in computed}
                data = {k: v for k, v in data.items() if k not in computed}
        obj = handler(data)
        for key, val in given.items():
            expected = to_jsonable_python(getattr(obj, key))
            if to_jsonable_python(val) != expected:
                raise ValueError(
                    f"{key!r} is computed for {cls.__name__} and cannot be set. "
                    f"Got {val!r}, but computed value is {expected!r}."
                )
        return obj

    @classmethod
    def _convert_renamed_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not (renamed := set(cls._renamed_fields).intersection(data)):
            return data
        data = dict(data)
        for old in renamed:
            new = cls._renamed_fields[old]
            if new in data:
                raise ValueError(
                    f"Cannot specify both {old!r} (deprecated) and {new!r} "
                    f"for {cls.__name__}."
                )
            warnings.warn(
                f"{cls.__name__}: {old!r} has been renamed to {new!r}. "
                f"Support for {old!r} will be removed in a future version.",
                FutureWarning,
                skip_file_prefixes=_SKIP_PREFIXES,
            )
            data[new] = data.pop(old)
        return data
