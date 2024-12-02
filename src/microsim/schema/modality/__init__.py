from ._simple_psf import Confocal, Identity, Widefield

Modality = Confocal | Widefield | Identity

__all__ = ["Confocal", "Identity", "Modality", "Widefield"]
