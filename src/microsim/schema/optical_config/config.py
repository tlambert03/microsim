import inspect
from collections.abc import Sequence
from typing import Any

from pydantic import Field, model_validator

from microsim._field_types import Watts, Watts_cm2
from microsim.fpbase import SpectrumOwner
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

from .filter import Filter, Placement, SpectrumFilter


class LightSource(SimBaseModel):
    name: str = ""
    spectrum: Spectrum
    power: Watts | Watts_cm2 | None = None

    @classmethod
    def from_fpbase(cls, light: SpectrumOwner) -> "LightSource":
        return cls(name=light.name, spectrum=Spectrum.from_fpbase(light.spectrum))

    def plot(self, show: bool = True) -> None:
        self.spectrum.plot(show=show)


class OpticalConfig(SimBaseModel):
    name: str = ""
    filters: list[Filter] = Field(default_factory=list)
    lights: list[LightSource] = Field(default_factory=list)

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
        exc = self.excitation
        if self.lights:
            l0, *rest = self.lights
            illum_spect = l0.spectrum
            if rest:
                for light in rest:
                    illum_spect = illum_spect * light.spectrum
            if exc:
                return illum_spect * exc.spectrum
            return illum_spect
        return exc.spectrum if exc else None

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
            ax.plot(filt.spectrum.wavelength.magnitude, filt.spectrum.intensity)
            legend.append(filt.name)
        if any(legend):
            ax.legend(legend)
        if show:
            plt.show()

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
