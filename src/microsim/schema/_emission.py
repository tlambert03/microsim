from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pint
from scipy.constants import Avogadro, c, h

if TYPE_CHECKING:
    from microsim.schema import Fluorophore, OpticalConfig, Spectrum

ureg = pint.application_registry.get()  # type: ignore
AVOGADRO = Avogadro / ureg.mol
PLANCK = h * ureg.joule * ureg.second
C = c * ureg.meter / ureg.second


def _ensure_quantity(value: Any, units: str, strict: bool = False) -> pint.Quantity:
    """Helper function to ensure that a value is a pint Quantity with `units`."""
    if isinstance(value, pint.Quantity):
        quant = value
    else:
        assert not strict, f"Expected a pint.Quantity with units {units}, got {value}"
        quant = ureg.Quantity(value)
    if quant.dimensionless:
        quant *= ureg.Quantity(units)
    _u = ureg.Quantity(units).units
    if not quant.is_compatible_with(_u):
        raise pint.DimensionalityError(quant.units, _u)
    return quant


def ec_to_cross_section(ec: Any) -> pint.Quantity:
    """Gives cross section in cm^2 from extinction coefficient in cm^2 * mol^-1."""
    ec = _ensure_quantity(ec, "cm^2/mol", strict=True)
    return (ec * np.log(10) / AVOGADRO).to("cm^2")  # type: ignore [no-any-return]


def energy_per_photon(wavelength: Any) -> pint.Quantity:
    """Converts a wavelength to energy per photon in J."""
    wavelength = _ensure_quantity(wavelength, "1nm")
    return (PLANCK * C / wavelength).to("J")  # type: ignore [no-any-return]


def get_excitation_rate(
    ex_filter_spectrum: Spectrum,
    fluor: Fluorophore,
    *,
    light_power: float = 100,
) -> Spectrum:
    """Calculate the number of emission events per fluorophore per second.

    This function takes an excitation spectrum configuration,
    and the name of a fluorophore, and returns the number of emission events per second
    per fluorophore.
    """
    # get the fluorophore
    if not (fluor_ex_spectrum := fluor.excitation_spectrum):
        raise NotImplementedError("Fluorophore has no excitation spectrum.")

    if (ext_coeff := fluor.extinction_coefficient) is None:
        ext_coeff = _ensure_quantity(55000, "cm^-1/M")
        warnings.warn(
            "No extinction coefficient provided for fluorophore, "
            "using 55,000 M^-1 * cm^-1.",
            stacklevel=2,
        )

    # TODO: derive light power from model
    irradiance = ex_filter_spectrum * _ensure_quantity(light_power, "W/cm^2")
    cross_section = fluor_ex_spectrum * ec_to_cross_section(ext_coeff)
    power_absorbed = cross_section * irradiance
    excitation_rate = power_absorbed / energy_per_photon(power_absorbed.wavelength)
    return excitation_rate

    # TODO
    # if the fluorophore has a lifetime, calculate the effective excitation rate
    # based on the fraction of time in the excited state
    # (i.e. at some point, we'll hit ground state depletion and saturation effects)
    if (lifetime := getattr(fluor, "lifetime", None)) is not None:
        lifetime = _ensure_quantity(lifetime, "ns")
        # Calculate the fraction of time in the excited state
        f_excited = excitation_rate * lifetime
        # Calculate the fraction of time in the ground state
        f_ground = 1 - f_excited
        # Calculate the effective excitation rate
        excitation_rate = excitation_rate * f_ground

    return excitation_rate


def get_emission_events(
    channel: OpticalConfig,
    fluor: Fluorophore,
    *,
    light_power: float = 100,
) -> Spectrum:
    # get the emission events for the given fluorophore
    if (channel_ex := channel.excitation) is None:
        raise NotImplementedError("Channel without excitation spectrum?")

    if not (fluor_em_spectrum := fluor.emission_spectrum):
        raise NotImplementedError("Fluorophore has no emission spectrum.")

    ex_rate = get_excitation_rate(channel_ex.spectrum, fluor, light_power=light_power)

    # convert the excitation rate to an emission rate, by matching the integral
    # of the excitation and emission spectra
    # I *think* this is correct, but I'm not 100% sure
    scaling_factor = ex_rate.integral() / fluor_em_spectrum.integral()
    fluor_em_rate = fluor_em_spectrum * scaling_factor

    # multiply by the quantum yield to get the number of emission events
    if fluor.quantum_yield is not None:
        fluor_em_rate = fluor_em_rate * fluor.quantum_yield

    # apply the emission filter(s)
    if channel_em := channel.emission:
        # TODO: see what (error) happens when there is no overlap between spectra
        fluor_em_rate = fluor_em_rate * channel_em.spectrum

    return fluor_em_rate
