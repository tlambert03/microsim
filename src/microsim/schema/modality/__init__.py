from ._simple import Confocal, Identity, Widefield

Modality = Confocal | Widefield | Identity

__all__ = ["Identity", "Modality", "Confocal", "Widefield"]
