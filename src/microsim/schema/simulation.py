import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import AfterValidator, Field, model_validator

from microsim._data_array import ArrayProtocol, from_cache, to_cache
from microsim.util import microsim_cache

from ._base_model import SimBaseModel
from .detectors import Detector
from .dimensions import Axis
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .optical_config import OpticalConfig, Placement
from .optical_config.lib import FITC
from .sample import FluorophoreDistribution, Sample
from .settings import Settings
from .space import ShapeScaleSpace, Space, _RelativeSpace
from .spectrum import Spectrum

if TYPE_CHECKING:
    from typing import Self, TypedDict, Unpack

    from .backend import NumpyAPI

    class SimulationKwargs(TypedDict, total=False):
        output_space: Space | dict | None
        objective_lens: ObjectiveLens
        channels: list[OpticalConfig]
        detector: Detector | None
        modality: Modality
        settings: Settings
        output_path: "OutPath" | None


def _check_extensions(path: Path) -> Path:
    if path.suffix not in {".tif", ".tiff", ".zarr", ".nc"}:
        raise ValueError("Recognized extensions include: .tif, .tiff, .zarr")
    return path


OutPath = Annotated[Path, AfterValidator(_check_extensions)]
xr.set_options(keep_attrs=True)  # type: ignore [no-untyped-call]


class Simulation(SimBaseModel):
    """Top level Simulation object."""

    truth_space: Space
    output_space: Space | None = None
    sample: Sample
    modality: Modality = Field(default_factory=Widefield)
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[OpticalConfig] = Field(default_factory=lambda: [FITC])
    # TODO: channels should also include `lights: list[LightSource]`
    detector: Detector | None = None
    settings: Settings = Field(default_factory=Settings)
    output_path: OutPath | None = None

    @classmethod
    def from_ground_truth(
        self,
        ground_truth: ArrayProtocol,
        scale: tuple[float, ...],
        **kwargs: "Unpack[SimulationKwargs]",
    ) -> "Self":
        """Shortcut to create a simulation directly from a ground truth array.

        In this case, we bypass derive the `truth_space` and `sample` objects directly
        from a pre-calculated ground truth array.  `scale` must also be provided as a
        tuple of floats, one for each dimension of the ground truth array.
        """
        return self(
            truth_space=ShapeScaleSpace(shape=ground_truth.shape, scale=scale),
            sample=Sample(labels=[FluorophoreDistribution.from_array(ground_truth)]),
            **kwargs,
        )

    @model_validator(mode="after")
    def _resolve_spaces(self) -> "Self":
        if isinstance(self.truth_space, _RelativeSpace):
            if self.output_space is not None:
                if isinstance(self.output_space, _RelativeSpace):
                    raise ValueError("Cannot have two relative spaces.")
                self.truth_space.reference = self.output_space
        elif isinstance(self.output_space, _RelativeSpace):
            self.output_space.reference = self.truth_space
        return self

    @property
    def _xp(self) -> "NumpyAPI":
        return self.settings.backend_module()

    def run(self, channels: int | Sequence[int] | None = None) -> xr.DataArray:
        """Run the simulation and return the result.

        This will also write a file to disk if `output` is set.
        """
        # truth = self.ground_truth()
        # if channels is None:
        #     channels = tuple(range(len(self.channels)))
        # elif isinstance(channels, int):
        #     channels = (channels,)
        # emission_flux = self.emission_flux(truth)
        optical_image = self.optical_image()
        image = self.digital_image(optical_image)
        self._write(image)
        return image

    def ground_truth(self) -> xr.DataArray:
        """Return the ground truth data.

        Returns position and quantity of fluorophores in the sample.  The return array
        has dimensions (F, Z, Y, X).  The units are fluorophores counts.

        Examples
        --------
        >>> sim = Simulation(...)
        >>> truth = sim.ground_truth()
        >>> truth.isel(f=0).max('z').plot()  # plot max projection of first fluorophore
        >>> plt.show()
        """
        if not hasattr(self, "_ground_truth"):
            xp = self._xp
            # make empty space into which we'll add the ground truth
            # TODO: this is wasteful... label.render should probably
            # accept the space object directly
            truth = self.truth_space.create(array_creator=xp.zeros)

            # render each ground truth
            label_data = []
            for label in self.sample.labels:
                cache_path = self._truth_cache_path(
                    label, self.truth_space, self.settings.random_seed
                )
                if self.settings.cache.read and cache_path and cache_path.exists():
                    data = from_cache(cache_path, xp=xp).astype(
                        self.settings.float_dtype
                    )
                    logging.info(
                        f"Loaded ground truth for {label} from cache: {cache_path}"
                    )
                else:
                    data = label.render(truth, xp=xp)
                    if self.settings.cache.write and cache_path:
                        to_cache(data, cache_path, dtype=np.uint16)

                label_data.append(data)

            # concat along the F axis
            fluors = [lbl.fluorophore for lbl in self.sample.labels]
            truth = xr.concat(label_data, dim=pd.Index(fluors, name=Axis.F))
            truth.attrs.update(units="fluorophores", long_name="Ground Truth")
            self._ground_truth = truth
        return self._ground_truth

    def filtered_emission_rates(self) -> xr.DataArray:
        """Return the emission rates for each fluorophore in each channel.

        Returns a (C, F, W) array of emission rates for each fluorophore in each
        channel, as a function of wavelength.  The units are photons/s.  The range
        of wavelengths will encompass the union of all the fluorophores' emission
        spectra, and the rates will be zero where the fluorophore does not emit.

        Examples
        --------
        >>> sim = Simulation(...)
        >>> rates = sim.filtered_emission_rates()
        >>> rates.isel(c=0).plot.line(x='w') # plot emission of all fluors in channel 0
        >>> plt.show()
        """
        qe = self.detector.qe if self.detector else None
        fluors = [lbl.fluorophore for lbl in self.sample.labels]
        nested_rates: list[list[xr.DataArray]] = [
            [oc.filtered_emission_rate(f, detector_qe=qe) for f in fluors]
            for oc in self.channels
        ]

        # combine xarray objects along the C and F axes, with outer join on W
        return xr.combine_nested(
            nested_rates,
            concat_dim=[Axis.C, Axis.F],
            combine_attrs="override",
            join="outer",
            fill_value=0,
        ).transpose(Axis.C, Axis.F, Axis.W)

    def optical_image(self) -> xr.DataArray:
        result = self.modality.render(
            self.ground_truth(),  # (F, Z, Y, X)
            self.filtered_emission_rates(),  # (C, F, W)
            objective_lens=self.objective_lens,
            settings=self.settings,
            xp=self._xp,
        ).sum(Axis.F)

        # Co-ordinates: (C, Z, Y, X)
        return result

    def emission_flux(self, truth: "xr.DataArray | None" = None) -> xr.DataArray:
        """Return the spatial emission in photons per second (after filters).

        This multiplies the per-fluorophore emission rates by the ground truth data to
        get the total emission flux for each voxel in the ground truth. The return
        array has dimensions (C, F, Z, Y, X).  The units are photons/s.

        Note, this integrates over all wavelengths (for reasons of memory efficiency).
        For finer control over the emission spectrum, you may wish to directly combine
        `filtered_emission_rates` with the ground truth data as needed.
        """
        if truth is None:
            truth = self.ground_truth()
        elif not isinstance(truth, xr.DataArray):
            raise ValueError("truth must be a DataArray")

        if Axis.F not in truth.coords:
            # we have no fluorophores to calculate
            return truth
        # total photons/s emitted by each fluorophore in each channel
        em_rates = self.filtered_emission_rates().sum(Axis.W)
        total_flux = em_rates * truth
        total_flux.attrs.update(unit="photon/sec", long_name="Emission Flux")
        # (C, F, Z, Y, X)
        return total_flux

    def _truth_cache_path(
        self,
        label: "FluorophoreDistribution",
        truth_space: "Space",
        seed: int | None,
    ) -> Path | None:
        if not (lbl_path := label.cache_path()):
            return None

        truth_cache = Path(microsim_cache("ground_truth"), *lbl_path)
        shape = f'shape{"_".join(str(x) for x in truth_space.shape)}'
        scale = f'scale{"_".join(str(x) for x in truth_space.scale)}'
        truth_cache = truth_cache / shape / scale
        if label.distribution.is_random():
            truth_cache = truth_cache / f"seed{seed}"
        return truth_cache

    def digital_image(
        self,
        optical_image: xr.DataArray | None = None,
        *,
        photons_pp_ps_max: int = 10000,
        exposure_ms: float = 100,
        with_detector_noise: bool = True,
    ) -> xr.DataArray:
        if optical_image is None:
            optical_image = self.optical_image()
        image = optical_image
        # TODO:Multi-fluorophore setup: combine information present in all wavelength
        # intervals.
        if self.output_space is not None:
            image = self.output_space.rescale(image)
        if self.detector is not None and with_detector_noise:
            im_max = self._xp.max(image.data)
            if not np.any(im_max):
                photon_flux = image
            else:
                photon_flux = image * photons_pp_ps_max / im_max
            gray_values = self.detector.render(
                photon_flux, exposure_ms=exposure_ms, xp=self._xp
            )
            image = gray_values
        image.attrs.update(unit="gray values")
        return image

    def _write(self, result: xr.DataArray) -> None:
        if not self.output_path:
            return
        if hasattr(result.data, "get"):
            result = result.copy(data=result.data.get(), deep=False)
        result.attrs["microsim.Simulation"] = self.model_dump_json()
        result.attrs.pop("space", None)
        result.coords[Axis.C] = [c.name for c in result.coords[Axis.C].values]
        if self.output_path.suffix == ".zarr":
            result.to_zarr(self.output_path, mode="w")
        elif self.output_path.suffix in (".nc",):
            result.to_netcdf(self.output_path)
        elif self.output_path.suffix in (".tif", ".tiff"):
            import tifffile as tf

            tf.imwrite(self.output_path, np.asanyarray(result))

    def plot(self, transpose: bool = False, legend: bool = True) -> None:
        plot_summary(self, transpose=transpose, legend=legend)


def plot_summary(sim: Simulation, transpose: bool = False, legend: bool = True) -> None:
    import matplotlib.pyplot as plt

    nrows = 5
    ncols = len(sim.channels)
    if transpose:
        nrows, ncols = ncols, nrows
    _fig, ax = plt.subplots(nrows, ncols, figsize=(18, 10), sharex=True)
    if transpose:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = ax.T
    else:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = ax
    if len(sim.channels) == 1:
        fp_ax, ex_ax, ab_ax, em_ax, f_ax = [fp_ax], [ex_ax], [ab_ax], [em_ax], [f_ax]

    for ch_idx, oc in enumerate(sim.channels):
        # FLUOROPHORES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                ex = fluor.absorption_cross_section
                ex.plot.line(ax=fp_ax[ch_idx], label=f"{fluor.name}")

        # ILLUMINATION PATH --------------------------------------
        ex_ax2 = ex_ax[ch_idx].twinx()
        for f in oc.filters:
            if f.placement == Placement.EM_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS:
                spect = spect.inverted()
            ex_ax2.plot(spect.wavelength, spect.intensity, label=f"{f.name}", alpha=0.4)
        # light sources
        for light in oc.lights:
            ls = light.spectrum
            ex_ax2.plot(ls.wavelength, ls.intensity, label=f"{light.name}", alpha=0.4)

        # combined illumination
        full = oc.illumination_flux_density
        full.plot.line(ax=ex_ax[ch_idx], label="flux density", color="k")

        # ABSORPTION/EMISSION RATES --------------------------------------
        for lbl in sim.sample.labels:
            if fluor := lbl.fluorophore:
                rate = oc.absorption_rate(fluor)
                tot = rate.sum()
                rate.isel({Axis.F: 0, Axis.C: 0}).plot.line(
                    ax=ab_ax[ch_idx],
                    x=Axis.W,
                    label=f"{fluor.name} ({tot:.2f} phot/s tot)",
                )

                em_rate = oc.total_emission_rate(fluor)
                em_rate.isel({Axis.F: 0, Axis.C: 0}).plot.line(
                    ax=ab_ax[ch_idx],
                    label=f"{fluor.name} emission",
                    alpha=0.4,
                    linestyle="--",
                )

        # EMISSION PATH --------------------------------------
        for f in oc.filters:
            if f.placement == Placement.EX_PATH:
                continue

            spect = f.spectrum
            if f.placement == Placement.BS_INV:
                spect = spect.inverted()
            em_ax[ch_idx].plot(
                spect.wavelength, spect.intensity, label=f"{f.name}", alpha=0.4
            )

        # detector
        if (detector := sim.detector) and (qe := detector.qe) is not None:
            kwargs = {
                "color": "gray",
                "label": f"{detector.name} QE",
                "linestyle": "--",
                "alpha": 0.4,
            }
            if isinstance(qe, Spectrum):
                em_ax[ch_idx].plot(qe.wavelength, qe.intensity, **kwargs)
            else:
                em_ax[ch_idx].axhline(qe, **kwargs)

        # combined emission/collection
        if ch_em := oc.emission:
            emspec = (ch_em.spectrum * qe).as_xarray()
            emspec.plot.line(ax=em_ax[ch_idx], label="emission", color="k")

            for lbl in sim.sample.labels:
                if fluor := lbl.fluorophore:
                    final = oc.filtered_emission_rate(fluor, detector_qe=qe)
                    final.isel({Axis.F: 0, Axis.C: 0}).plot.line(
                        ax=f_ax[ch_idx],
                        label=f"{fluor.name} collection ({final.sum():.2f} phot/s tot)",
                    )

        if legend:
            fp_ax[ch_idx].legend(loc="upper right")
            ex_ax2.legend(loc="upper right")
            ab_ax[ch_idx].legend(loc="upper right")
            f_ax[ch_idx].legend()
            em_ax[ch_idx].legend()
            # oc_ax[ch_idx].legend(loc="right")

        # LABELS --------------------------------------
        ex_ax[ch_idx].set_title(oc.name)
        fp_ax[ch_idx].set_xlabel("")
        ex_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].set_xlabel("")
        ab_ax[ch_idx].set_title("")
        ab_ax[ch_idx].set_ylabel("[photons/s]")
        em_ax[ch_idx].set_xlabel("")
        f_ax[ch_idx].set_xlabel("wavelength [nm]")
        f_ax[ch_idx].set_ylabel("[photons/s]")
        f_ax[ch_idx].set_title("")

    fp_ax[0].set_xlim(400, 750)  # shared x-axis
    plt.tight_layout()
    plt.show()
