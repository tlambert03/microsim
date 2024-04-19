from .confocal import Confocal
from .sim import SIM2D, SIM3D
from .widefield import Widefield

Modality = Confocal | Widefield | SIM2D | SIM3D


__all__ = ["Modality", "Confocal", "Widefield", "SIM2D", "SIM3D"]
