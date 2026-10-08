from ._simple_psf import Confocal, Identity, SpinningDiskConfocal, Widefield

Modality = Confocal | SpinningDiskConfocal | Widefield | Identity

__all__ = ["Confocal", "Identity", "Modality", "SpinningDiskConfocal", "Widefield"]
