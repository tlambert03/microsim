from .confocal import Confocal
from .widefield import Widefield

Modality = Confocal | Widefield


__all__ = ["Modality", "Confocal", "Widefield"]
