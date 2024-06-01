from ._simple import Confocal, Widefield

Modality = Confocal | Widefield

__all__ = ["Modality", "Confocal", "Widefield"]
