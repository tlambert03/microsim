from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from microsim._data_array import xrDataArray
    from microsim.schema.backend import NumpyAPI


class _BaseDistribution(BaseModel, ABC):
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
