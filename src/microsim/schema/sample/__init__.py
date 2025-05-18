from typing import TYPE_CHECKING, Any

from ._distributions._base import BaseDistribution
from ._distributions.matslines import MatsLines
from .fluorophore import Fluorophore
from .sample import AnyDistribution, FluorophoreDistribution, Sample

if TYPE_CHECKING:
    from ._distributions.cosem import CosemLabel


__all__ = [
    "AnyDistribution",
    "BaseDistribution",
    "CosemLabel",
    "Fluorophore",
    "FluorophoreDistribution",
    "MatsLines",
    "Sample",
]


def __getattr__(name: str) -> Any:
    if name == "CosemLabel":
        from ._distributions.cosem import CosemLabel

        return CosemLabel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
