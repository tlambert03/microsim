from microsim._data_array import DataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class ExcitationModel(SimBaseModel):
    name: str
    excitation_spectrum: Spectrum
    incoming_light_spectrum: Spectrum
    duration: float
    """
    The excitation model simulates excitation of fluorophores by the light.
    This should be instantiated for each fluorophore.
    It is defined by the excitation spectrum, the spatial density of the fluorophores
    in the sample and the duration of the excitation. Sampling
    from Poisson distribution happens here.
    """

    def sample(self, fluorophore_spatial_density: DataArray) -> DataArray:
        # TODO: Multi-fluorophore setup: model the excitation.
        raise NotImplementedError("Needs to be implemented")


class EmissionModel(SimBaseModel):
    """The fluorophore emission model.

    It is essentially selecting the wavelength interval for the
    emitted photons. In code, this will be equivalent to distributing the pixelwise
    intensity to the different emmision wavelength intervals. This class will be
    instantiated for each fluorophore.
    """

    emission_spectrum: Spectrum

    def attach_emission_spectra(self, emitted_intensity: DataArray) -> DataArray:
        raise NotImplementedError("Needs to be implemented")


class ExcitationEmissionModel(SimBaseModel):
    """The combined excitation-emission model.

    It is a model for the light that excites the fluorophores and the emits light.
    This class will be instantiated for each fluorophore.
    """

    excitation: ExcitationModel
    emission: EmissionModel

    def sample(self, fluorophore_spatial_density: DataArray) -> DataArray:
        sample = self.excitation.sample(fluorophore_spatial_density)
        return self.emission.attach_emission_spectra(sample)
