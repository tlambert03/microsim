from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum
from microsim._data_array import DataArray


class ExcitationModel(SimBaseModel):
    name: str
    excitation_spectrum: Spectrum
    incoming_light_spectrum: Spectrum
    duration: float
    """
    The excitation model simulates excitation of fluorophores by the light. This should be instantiated for each fluorophore
    It is defined by the excitation spectrum, the spatial density of the fluorophores in the sample and the duration of the excitation. Sampling 
    from Poisson distribution happens here.
    """


    def sample(self, fluorophore_spatial_density: DataArray
):
        # TODO: Multi-fluorophore setup: model the excitation.  
        pass


class EmissionModel(SimBaseModel):
    """
    The emission model is essentially selecting the wavelength bin for the emitted photons. This can be instantiated for each fluorophore.
    """
    emission_spectrum: Spectrum

    def attach_emission_spectra(self, emitted_intensity: DataArray):
        pass


class ExcitationEmissionModel(SimBaseModel):
    """
    The excitation-emission model is a model for the light that excites the fluorophores and the emits light. This can be instantiated for each fluorophore.
    """
    excitation: ExcitationModel
    emission: EmissionModel

    def sample(self)->DataArray:
        sample = self.excitation.sample()
        return self.emission.attach_emission_spectra(sample)
