import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import xarray as xr
from pydantic import Field, model_validator
from scipy.constants import c, h

from microsim.fpbase import SpectrumOwner
from microsim.schema._base_model import SimBaseModel
from microsim.schema.detectors import Detector
from microsim.schema.dimensions import Axis
from microsim.schema.sample.fluorophore import Fluorophore
from microsim.schema.spectrum import Spectrum

from .filter import Filter, Placement, SpectrumFilter

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class LightSource(SimBaseModel):
    name: str = ""
    spectrum: Spectrum
    power: float | None = None  # W/cm^2

    @classmethod
    def from_fpbase(cls, light: SpectrumOwner) -> "LightSource":
        return cls(name=light.name, spectrum=Spectrum.from_fpbase(light.spectrum))

    def plot(self, show: bool = True) -> None:
        self.spectrum.plot(show=show)  # type: ignore [call-arg]

    @classmethod
    def laser(cls, wavelength: float, power: float | None = None) -> "LightSource":
        return cls(
            name=f"{wavelength}nm Laser",
            spectrum=Spectrum(
                wavelength=[wavelength - 1, wavelength, wavelength + 1],
                intensity=[0, 1, 0],
            ),
            power=power,
        )


class OpticalConfig(SimBaseModel):
    name: str = ""
    filters: list[Filter] = Field(default_factory=list)
    lights: list[LightSource] = Field(default_factory=list)
    detector: Detector | None = None
    exposure_ms: float | None = None

    # seemingly duplicate of power in LightSource
    # but it depends on where the power is being measured
    # TODO: it's tough deciding where power should go...
    # it could also go on Simulation itself as a function of space.
    power: float | None = None  # total power of all lights after filters

    def absorption_rate(self, fluorophore: Fluorophore) -> xr.DataArray:
        """Return the absorption rate of a fluorophore with this configuration.

        The absorption rate is the number of photons absorbed per second per
        fluorophore, as a function of wavelength. It's a vector with a single axis W,
        and singleton dimensions F and C.
        """
        illum_flux_density = self.illumination_flux_density  # photons/cm^2/s
        cross_section = fluorophore.absorption_cross_section  # cm^2/fluorophore

        # absorption rate in photons/s/fluorophore
        abs_rate = illum_flux_density * cross_section

        # add singleton coordinates for fluorophore and self
        abs_rate = abs_rate.expand_dims({Axis.F: [fluorophore], Axis.C: [self]})

        # add metadata
        abs_rate.name = "absorption_rate"
        abs_rate.attrs["long_name"] = "Absorption rate"
        abs_rate.attrs["units"] = "photons/s"
        return abs_rate

    def total_emission_rate(self, fluorophore: Fluorophore) -> xr.DataArray:
        """Return the emission rate of a fluorophore with this configuration.

        The emission rate is the total number of photons emitted per second per
        fluorophore, as a function of wavelength, prior to any filtering in the emission
        path. It's a vector with a single axis W, and singleton dimensions F and C
        """
        tot_absorption_rate = self.absorption_rate(fluorophore).sum()
        em_rate = fluorophore.emission_spectrum.as_xarray()
        # norm area to 1
        em_rate = em_rate / em_rate.sum()
        # multiply by quantum yield and total absorption rate
        em_rate = em_rate * (fluorophore.quantum_yield or 1) * tot_absorption_rate
        # add singleton coordinates for fluorophore and self
        em_rate = em_rate.expand_dims({Axis.F: [fluorophore], Axis.C: [self]})
        em_rate.name = "emission_rate"
        em_rate.attrs["long_name"] = "Emission rate"
        em_rate.attrs["units"] = "photons/s"
        return em_rate

    def filtered_emission_rate(
        self, fluorophore: Fluorophore, detector_qe: float | Spectrum | None = None
    ) -> xr.DataArray:
        """Return the emission rate of a fluorophore with this config, after filters.

        The emission rate is the number of photons emitted per second per
        fluorophore, as a function of wavelength, after being excited by this optical
        config, then passing through the emission path of this config. It's a vector
        with a single axis W, and singleton dimensions F and C.

        This is the complete picture of the treatment of a specific fluorophore with
        this optical configuration.  It takes into account:

            - the excitation spectrum and extinction coefficient of the fluorophore
            - the excitation filter/beamsplitter and light source spectra
            - the quantum yield and emission spectrum of the fluorophore
            - the emission filter/beamsplitter spectra
            - camera QE, if passed
        """
        if self.emission is None:
            em_array: xr.DataArray | float = 1.0
        else:
            em_spectrum = self.emission.spectrum
            if detector_qe is None and self.detector is not None:
                detector_qe = self.detector.qe
            if detector_qe is not None:
                em_spectrum = em_spectrum * detector_qe
            em_array = em_spectrum.as_xarray()

        final = self.total_emission_rate(fluorophore) * em_array
        final.name = "filtered_emission_rate"
        final.attrs["long_name"] = "Filtered Emission rate"
        final.attrs["units"] = "photons/s"
        return final

    @property
    def excitation(self) -> Filter | None:
        """Combine all excitation filters into a single spectrum."""
        filters = []
        for f in self.filters:
            if f.placement in {Placement.EX_PATH, Placement.BS_INV, Placement.ALL}:
                filters.append(f)
            if f.placement == Placement.BS:
                filters.append(f.inverted())
        return self._merge(filters, spectrum="excitation")

    @property
    def illumination(self) -> Spectrum | None:
        """Return the combined illumination spectrum.

        This represents the spectrum of light source and all of the filters in the
        excitation path. If there are multiple light sources, they are combined.
        """
        exc = self.excitation
        if self.lights:
            l0, *rest = self.lights
            illum_spect = l0.spectrum
            if rest:
                for light in rest:
                    illum_spect = illum_spect + light.spectrum
            if exc:
                return illum_spect * exc.spectrum
            return illum_spect
        return exc.spectrum if exc else None

    @property
    def irradiance(self) -> xr.DataArray:
        """Return the illumination irradiance in W/cm^2.

        This scales the illumination spectrum to power. It is a measure of the power per
        unit area of the excitation path of this optical configuration, in W/cm^2, as
        a function of wavelength.
        """
        if (illum := self.illumination) is None:
            raise ValueError("This Optical Config has no illumination spectrum.")
        # get irradiance scaled to power
        irrad = illum.as_xarray()  # W/cm^2
        # normalize area under curve to 1
        irrad = irrad / irrad.sum()
        # scale to power
        if self.power is not None:
            irrad = irrad * self.power
        irrad.name = "irradiance"
        irrad.attrs["long_name"] = "Irradiance"
        irrad.attrs["units"] = "W/cm^2"
        return irrad

    @property
    def illumination_flux_density(self) -> xr.DataArray:
        """Return the illumination flux density in photons/cm^2/s.

        This is a measure of the number of photons per unit area per second in the
        excitation path of this optical configuration.
        It converts irradiance (W/cm^2) to photons/cm^2/s using the energy of a photon.
        """
        # get irradiance scaled to power
        irrad = self.irradiance  # W/cm^2
        # convert W/cm^2 to photons/cm^2/s using E = h * c / λ
        wavelength_meters = cast("xr.DataArray", irrad.coords["w"] * 1e-9)
        joules_per_photon = h * c / wavelength_meters
        illum_flux_density = irrad / joules_per_photon  # photons/cm^2/s

        illum_flux_density.name = "illum_flux_density"
        illum_flux_density.attrs["long_name"] = "Illuminance Flux Density"
        illum_flux_density.attrs["units"] = "photons/cm^2/s"

        return cast("xr.DataArray", illum_flux_density)

    @property
    def emission(self) -> Filter | None:
        """Combine all emission filters into a single spectrum."""
        filters = []
        for f in self.filters:
            if f.placement in {Placement.EM_PATH, Placement.BS, Placement.ALL}:
                filters.append(f)
            if f.placement == Placement.BS_INV:
                filters.append(f.inverted())
        return self._merge(filters, spectrum="emission")

    def _merge(
        self, filters: Sequence[Filter], spectrum: str = "spectrum"
    ) -> Filter | None:
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        effective_spectrum = filters[0].spectrum
        for filt in filters[1:]:
            effective_spectrum = effective_spectrum * filt.spectrum
        return SpectrumFilter(
            name=f"Effective {spectrum} for {self.name}",
            transmission=effective_spectrum,
        )

    @classmethod
    def from_fpbase(
        cls, microscope_id: str, config_name: str | None = None
    ) -> "OpticalConfig":
        from microsim.fpbase import get_microscope

        if config_name is None:
            if "::" not in microscope_id:  # pragma: no cover
                raise ValueError(
                    "If config_name is not provided, microscope_id must be "
                    "in the form 'scope::config'"
                )
            microscope_id, config_name = microscope_id.split("::")

        fpbase_scope = get_microscope(microscope_id)
        for cfg in fpbase_scope.opticalConfigs:
            if cfg.name.lower() == config_name.lower():
                if cfg.light:
                    lights = [LightSource.from_fpbase(cfg.light)]
                else:
                    lights = []
                return cls(
                    name=cfg.name,
                    filters=[SpectrumFilter.from_fpbase(f) for f in cfg.filters],
                    lights=lights,
                )

        raise ValueError(
            f"Could not find config named {config_name!r} in FPbase microscope "
            f"{microscope_id!r}. Available names: "
            f"{', '.join(repr(c.name) for c in fpbase_scope.opticalConfigs)}"
        )

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, str):
            if "::" not in value:  # pragma: no cover
                raise ValueError(
                    "If OpticalConfig is provided as a string, it must be "
                    "in the form 'fpbase_scope_id::config_name'"
                )
            # TODO: seems weird to have to cast back to dict...
            # but otherwise doesn't work with 'before' validator.  look into it.
            return cls.from_fpbase(value).model_dump()
        return value

    def plot(self, show: bool = True) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)

        legend = []
        for filt in self.filters:
            ax.plot(filt.spectrum.wavelength, filt.spectrum.intensity)
            legend.append(filt.name)
        if any(legend):
            ax.legend(legend)
        if show:
            plt.show()

    def all_spectra(self) -> xr.DataArray:
        """Return a DataArray with all spectra in this configuration."""
        data, coords = [], []
        for filt in self.filters:
            data.append(filt.spectrum.as_xarray())
            coords.append(f"{filt.name} ({filt.placement.name})")
        for light in self.lights:
            data.append(light.spectrum.as_xarray())
            coords.append(light.name)
        da: xr.DataArray = xr.concat(data, dim="spectra")
        da.coords.update({"spectra": coords})
        return da

    # WARNING: dark magic ahead
    # This is a hack to make OpticalConfig hashable and comparable, but only
    # when used in the context of a pandas DataFrame or xarray DataArray coordinate.
    # this allows syntax like `data_array.sel(c='FITC')` to work as expected.
    def __hash__(self) -> int:
        frame = inspect.stack()[1]
        if "pandas" in frame.filename and frame.function == "get_loc":
            return hash(self.name)
        return id(self)

    def __eq__(self, value: object) -> bool:
        frame = inspect.stack()[1]
        if "pandas" in frame.filename and frame.function == "get_loc":
            return hash(self.name) == hash(value)
        return super().__eq__(value)

    def __str__(self) -> str:
        return self.name

    def plot_excitation(self, ax: "Axes | None" = None) -> None:
        """Plot all components of the excitation path."""
        # combined illumination
        lines = []
        labels = ["combined flux"]

        # plot the combined illumination spectrum
        full = self.illumination_flux_density
        lines.extend(full.plot.line(ax=ax, color="k", linewidth=2))

        # plot individual filters
        ax0 = cast("Axes", lines[0].axes)
        ax2 = ax0.twinx()
        for f in self.filters:
            if f.placement == Placement.EM_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS:
                spect = spect.inverted()
            lines.extend(spect.plot(ax=ax2, alpha=0.6))  # type: ignore [call-arg]
            labels.append(f.name)

        # light sources
        for light in self.lights:
            lines.extend(light.spectrum.plot(ax=ax2, alpha=0.6))  # type: ignore [call-arg]
            labels.append(light.name)

        # formatting
        ax2.set_ylabel("")
        ax2.yaxis.set_ticks_position("none")
        ax2.set_yticklabels([])
        ax2.set_xlim(400, 700)
        ax0.set_title(self.name)
        ax0.legend(
            lines,
            labels,
            ncols=2,
            fontsize="small",
            loc="lower right",
            bbox_to_anchor=(1, 1.1),
        )

    def plot_emission(
        self, ax: "Axes | None" = None, detector_qe: float | Spectrum | None = None
    ) -> None:
        """Plot all components of the emission path."""
        # plot individual filters
        alpha = 0.5
        for f in self.filters:
            if f.placement == Placement.EX_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS_INV:
                spect = spect.inverted()
            [line] = spect.plot(ax=ax, alpha=alpha, label=f.name)  # type: ignore [call-arg]

        ax = cast("Axes", line.axes)
        kwargs = {"color": "gray", "label": "QE", "linestyle": "--", "alpha": alpha}
        if isinstance(detector_qe, Spectrum):
            detector_qe.plot(ax=ax, **kwargs)  # type: ignore
        elif isinstance(detector_qe, int | float):
            ax.axhline(detector_qe, **kwargs)

        # combined
        if (emission := self.emission) is not None:
            em_spectrum = emission.spectrum.as_xarray()
            if detector_qe is not None:
                em_spectrum = em_spectrum * detector_qe
            em_spectrum.plot.line(ax=ax, color="k", linewidth=2, label="combined")
        ax.legend()
        ax.set_xlim(400, 800)
        ax.set_xlabel("transmission [%]")
