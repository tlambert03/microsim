from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import runtime_checkable

if TYPE_CHECKING:
    from microsim._data_array import xrDataArray
    from microsim.schema.backend import NumpyAPI


@runtime_checkable
class Renderable(Protocol):
    @abstractmethod
    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        """Render the distribution into the given space."""


class _IsInstanceAnySer:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _validate(obj: Any) -> Any:
            if not isinstance(obj, source_type):
                raise ValueError(f"Expected {source_type}, got {type(obj)}")
            return obj

        return core_schema.no_info_before_validator_function(
            _validate, core_schema.any_schema()
        )


RenderableType = Annotated[Renderable, _IsInstanceAnySer()]


class BaseDistribution(BaseModel, ABC):
    @classmethod
    def is_random(cls) -> bool:
        """Return True if this distribution generates randomized results."""
        return True

    def cache_path(self) -> tuple[str, ...] | None:
        """Return a tuple of strings that uniquely identify this distribution.

        This may be used to determine a cache location for ground truths generated
        by this distribution. If None is returned, no caching will be performed.
        """
        return None

    @abstractmethod
    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        """Render the distribution into the given space."""
        raise NotImplementedError
