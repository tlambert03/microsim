from ._simple import Confocal, Widefield, Identity

Modality = Confocal | Widefield | Identity

__all__ = ["Identity", "Modality", "Confocal", "Widefield"]
