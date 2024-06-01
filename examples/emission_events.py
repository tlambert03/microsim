from typing import Any

import numpy as np
import pint
from scipy.constants import Avogadro, c, h

from microsim.fpbase import get_fluorophore, get_microscope

ureg = pint.application_registry.get()
AVOGADRO = Avogadro / ureg.mol
PLANCK = h * ureg.joule * ureg.second
C = c * ureg.meter / ureg.second


def get_overlapping_spectra(
    spectra: np.ndarray, spectrum2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping subset of spectra.

    This assumes that the spectra are (N, 2) arrays where the first column is the
    wavelength and the second column is the intensity, and that the spectra are
    sorted by wavelength. (This is the format of the data returned by FPbase.)
    """
    # Find the indices of the start and end of the overlapping subset
    start = max(spectra[0, 0], spectrum2[0, 0])
    end = min(spectra[-1, 0], spectrum2[-1, 0])

    # Find the indices of the start and end of the overlapping subset
    start_idx = np.searchsorted(spectra[:, 0], start)
    end_idx = np.searchsorted(spectra[:, 0], end, side="right")

    start_idx2 = np.searchsorted(spectrum2[:, 0], start)
    end_idx2 = np.searchsorted(spectrum2[:, 0], end, side="right")
    return spectra[start_idx:end_idx], spectrum2[start_idx2:end_idx2]


def _ensure_quantity(value: Any, units: str) -> pint.Quantity:
    """Helper function to ensure that a value is a pint Quantity with `units`."""
    if isinstance(value, pint.Quantity):
        quant = value
    else:
        quant = ureg.Quantity(value)
    if quant.dimensionless:
        quant *= ureg.Quantity(units)
    _u = ureg.Quantity(units).units
    if not quant.is_compatible_with(_u):
        raise pint.DimensionalityError(quant.units, _u)
    return quant


def ec_to_cross_section(ec: Any) -> pint.Quantity:
    """Gives cross section in cm^2 from extinction coefficient in M^-1 * cm^-1."""
    ec = _ensure_quantity(ec, "cm^2/mol")
    # x1000?
    # this came from calculations elsewhere, and looking at wikipedia
    # and looking at Nathan Shaner's code
    # need to double check whether it's still correct with our units
    ec = ec * 1000
    return (ec * np.log(10) / AVOGADRO).to("cm^2")


def energy_per_photon(wavelength: Any) -> pint.Quantity:
    """Converts a wavelength to energy per photon in J."""
    wavelength = _ensure_quantity(wavelength, "1nm")
    return (PLANCK * C / wavelength).to("J")


def fluorophore_photon_flux(
    wavelength: Any,
    irradiance: Any,
    extinction_coefficient: Any,
) -> pint.Quantity:
    """Calculates the number of photons hitting a fluorophore per second."""
    irradiance = _ensure_quantity(irradiance, "1W/cm^2")
    E_photon = energy_per_photon(wavelength)
    cross_section = ec_to_cross_section(extinction_coefficient)
    exc_rate = (cross_section * irradiance / E_photon).to("1/s")

    return exc_rate


def get_emission_events(
    microscope_id: str,
    optical_config: str,
    fluorophore: str,
    *,
    light_power: float = 100,
) -> np.ndarray:
    """Calculate the number of emission events per fluorophore per second.

    This function takes an FPbase microscope ID, the name of an optical configuration,
    and the name of a fluorophore, and returns the number of emission events per second
    per fluorophore.
    """
    # get the microscope and fluorophore
    microscope = get_microscope(microscope_id)

    # find the excitation filter in the microscope
    try:
        WF_Green = next(
            oc for oc in microscope.opticalConfigs if oc.name == optical_config
        )
    except StopIteration as e:
        raise ValueError(
            f"Optical config {optical_config} not found in microscope {microscope_id}"
        ) from e

    try:
        ex_filter = next(f for f in WF_Green.filters if f.spectrum.subtype == "BP")
    except StopIteration as e:
        raise ValueError(
            f"Bandpass filter not found in optical config {optical_config}"
        ) from e

    # get the fluorophore
    fluor = get_fluorophore(fluorophore).default_state
    if not fluor:
        raise ValueError(f"Fluorophore {fluorophore} has no default state.")
    if not fluor.excitation_spectrum:
        raise ValueError(f"Fluorophore {fluorophore} has no excitation spectrum.")

    # convert the spectra to numpy arrays
    fluor_ex_spectrum = np.asarray(fluor.excitation_spectrum.data)
    filter_spectrum = np.asarray(ex_filter.spectrum.data)

    # find the subset of the spectra with overlapping wavelengths
    ex_spectrum, filter_spectrum = get_overlapping_spectra(
        fluor_ex_spectrum, filter_spectrum
    )

    # wavelengths in nm
    wavelengths = filter_spectrum[:, 0] * ureg.nm
    # power in units of W x cm-2 (irradiance), we're making up constant 100 for now...
    irradiance = filter_spectrum[:, 1] * _ensure_quantity(light_power, "W/cm^2")
    # scale ex_spectrum by extinction coefficient
    # note, fluor.extCoeff is already a pint Quantity of units 1/M/cm
    ext_coeff = ex_spectrum[:, 1] * fluor.extCoeff

    # calculate the number of photons hitting a fluorophore per second
    exc_rate = fluorophore_photon_flux(wavelengths, irradiance, ext_coeff)

    # if the fluorophore has a lifetime, calculate the effective excitation rate
    # based on the fraction of time in the excited state
    # (i.e. at some point, we'll hit ground state depletion and saturation effects)
    if (lifetime := getattr(fluor, "lifetime", None)) is not None:
        lifetime = _ensure_quantity(lifetime, "ns")
        # Calculate the fraction of time in the excited state
        f_excited = exc_rate * lifetime
        # Calculate the fraction of time in the ground state
        f_ground = 1 - f_excited
        # Calculate the effective excitation rate
        exc_rate = exc_rate * f_ground

    # multiply by the quantum yield to get the number of emission events
    emission_events = exc_rate * fluor.qy

    # recombine with wavelength:
    return wavelengths, emission_events


# we'll be calculating this page:
# https://www.fpbase.org/microscope/wKqWb/?c=Widefield%20Green&p=egfp_default
wavelengths, emission_events_per_fluor = get_emission_events(
    "wKqWb", "Widefield Green", "EGFP"
)
print(max(emission_events_per_fluor))
