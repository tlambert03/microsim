from microsim.schema._base_model import SimBaseModel
from typing import Literal

MODALITIES = ["widefield", "confocal", "two-photon", "light-sheet"]

class MicroscopeModality(SimBaseModel):
    type: Literal[MODALITIES]

    def generate_psf(self, space, ex_wvl_um, em_wvl_um, objective_lens, settings, xp):
        raise NotImplementedError('Must be implemented in subclasses')
    
    def get_psfs(self, space, channel, objective_lens, settings, xp):
        # TODO: the function to get multiple psfs from the channel
        # It will call the modality specific generate_psf function.
        pass

    def render(self, truth, channel, objective_lens, settings, xp):
        psfs = self.get_psfs(truth, channel, objective_lens, settings, xp)
        imgs = [xp.fftconvolve(truth.data, em_psf, mode="same") for em_psf in psfs]
        # TODO: superposition logic.
        # return dataarray.
        pass