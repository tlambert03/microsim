import time
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import (
    AfterValidator,
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from microsim._data_array import ArrayProtocol, from_cache, to_cache
from microsim._logger import logger, logging_indented
from microsim.util import microsim_cache

from ._base_model import SimBaseModel
from .detectors import Detector
from .dimensions import Axis
from .lens import ObjectiveLens
from .modality import Modality, Widefield
from .optical_config import OpticalConfig, Placement
from .optical_config._fluorescence import (
    ChannelFluorescence,
    track_absorption_rate,
    track_light_sources,
)
from .optical_config.lib import FITC
from .sample import Fluorophore, FluorophoreDistribution, Sample
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
        tracks: "Tracks"
        output_path: "OutPath" | None


def _check_extensions(path: Path) -> Path:
    if path.suffix not in {".tif", ".tiff", ".zarr", ".nc"}:
        raise ValueError("Recognized extensions include: .tif, .tiff, .zarr")
    return path


OutPath = Annotated[Path, AfterValidator(_check_extensions)]


class Track(SimBaseModel):
    """Channels acquired at the same time, as on a point scanner.

    All light sources of all channels in a track are on, and each channel's detector
    records at once.  May also be given as a bare list of channels, e.g. `[0, 2]`.
    """

    channels: list[StrictInt | str] = Field(
        min_length=1,
        description="Channels in this track: by index in `Simulation.channels`, or "
        "by channel name.",
    )

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, list | tuple):
            return {"channels": value}
        return value


Tracks = Literal["sequential", "simultaneous"] | list[Track]
_TRACKS_HINT = (
    "  (To change both `channels` and `tracks`, first set `tracks='sequential'`.)"
)
xr.set_options(keep_attrs=True)  # type: ignore [no-untyped-call]

# LRU cache of `Simulation._channel_fluorescence` results (small C x F x W arrays)
_FLUORESCENCE_CACHE: OrderedDict[tuple, list[list[ChannelFluorescence]]] = OrderedDict()
_FLUORESCENCE_CACHE_SIZE = 32


class Simulation(SimBaseModel):
    """Top level Simulation object."""

    truth_space: Space
    output_space: Space | None = None
    sample: Sample
    modality: Modality = Field(default_factory=Widefield)
    objective_lens: ObjectiveLens = Field(default_factory=ObjectiveLens)
    channels: list[OpticalConfig] = Field(default_factory=lambda: [FITC])
    # TODO: channels should also include `lights: list[LightSource]`
    detector: Detector | None = Field(default=None, discriminator="camera_type")
    # Per-pixel integration time: camera exposure, or pixel dwell time for
    # point-scanning modalities (e.g. Confocal). May be renamed `integration_time`.
    exposure_ms: float = 100
    tracks: Tracks = Field(
        default="sequential",
        description="Which channels are acquired at the same time (a *track*, as on "
        "a point scanner): all light sources of all channels in a track are on, and "
        "each channel's detector records at once.  `'sequential'` (the default): "
        "each channel is its own track, excited only by its own light sources.  "
        "`'simultaneous'`: one track with all channels.  Or a list of tracks, each a "
        "`Track` or a list of channels given by index in `channels` or by name, "
        "e.g. `[[0, 2], [1, 3]]`.  Every channel must be in exactly one track.",
    )
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

    @model_validator(mode="after")
    def _validate_tracks(self) -> "Self":
        point_scanning = not self.modality.detection_type.sums_over_pixel_area
        for track in self._track_indices():
            channels = [self.channels[c] for c in track]
            track_light_sources(channels)  # checks shared light sources
            dwells = {oc.exposure_ms or self.exposure_ms for oc in channels}
            if point_scanning and len(dwells) > 1:
                raise ValueError(
                    f"Channels {[oc.name for oc in channels]} are acquired together "
                    "(in the same track), so they must have the same pixel dwell time "
                    f"(`exposure_ms`); got {sorted(dwells)}."
                )
        return self

    def _track_indices(self) -> list[list[int]]:
        """Return `tracks` as lists of channel indices."""
        n = len(self.channels)
        if self.tracks == "sequential":
            return [[i] for i in range(n)]
        if self.tracks == "simultaneous":
            return [list(range(n))]
        names = [oc.name for oc in self.channels]
        tracks: list[list[int]] = []
        for track in self.tracks:
            indices = []
            for ch in track.channels:
                if isinstance(ch, str):
                    if names.count(ch) != 1:
                        found = "matches several" if ch in names else "matches no"
                        raise ValueError(
                            f"Track channel {ch!r} {found} channel names "
                            f"({names}). Use channel indices instead."
                        )
                    ch = names.index(ch)
                elif not 0 <= ch < n:
                    raise ValueError(
                        f"Track channel index {ch} is out of range.{_TRACKS_HINT}"
                    )
                indices.append(ch)
            tracks.append(indices)
        listed = sorted(i for track in tracks for i in track)
        if listed != list(range(n)):
            raise ValueError(
                "Every channel must be in exactly one track; got tracks "
                f"{tracks} for {n} channels.{_TRACKS_HINT}"
            )
        return tracks

    def __setattr__(self, name: str, value: Any) -> None:
        old = vars(self).get(name)
        was_set = name in self.model_fields_set
        try:
            super().__setattr__(name, value)
        except ValidationError:
            # model validators run after the new value is set: restore the old one
            if name in type(self).model_fields:
                vars(self)[name] = old
                if not was_set:
                    self.model_fields_set.discard(name)
            raise
        if name in type(self).model_fields:
            # any field may affect the ground truth (sample, spaces, settings...)
            vars(self).pop("_ground_truth", None)

    if not TYPE_CHECKING:

        def model_copy(
            self, *, update: Mapping[str, Any] | None = None, deep: bool = False
        ) -> "Self":
            """Copy the simulation, applying (and validating) any `update` fields.

            Unlike pydantic's default, the updated simulation is validated as a
            whole (so relative spaces are re-resolved), and the cached ground truth
            is dropped.
            """
            copied = super().model_copy(deep=deep)
            # `_resolve_spaces` sets the `reference` of a relative space, which a
            # shallow copy shares with `self`: give the copy its own.
            for name in ("truth_space", "output_space"):
                if isinstance(space := getattr(copied, name), _RelativeSpace):
                    vars(copied)[name] = space.model_copy()
            if not update:
                return copied

            # validate all fields at once (not one at a time), since intermediate
            # states may be invalid (e.g. swapping which space is relative)
            fields = {name: getattr(copied, name) for name in type(self).model_fields}
            new = type(self).model_validate({**fields, **update})
            new.__pydantic_fields_set__ = copied.model_fields_set | set(update)
            return new

    @property
    def _xp(self) -> "NumpyAPI":
        return self.settings.backend_module()

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
        logger.info(f"Creating ground truth (shape ={self.truth_space.shape}) + ")
        _t0 = time.perf_counter()

        if not hasattr(self, "_ground_truth"):
            with logging_indented():
                xp = self._xp

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
                        logger.info(
                            f"Loaded ground truth for {label} from cache: {cache_path}"
                        )
                    else:
                        # make empty space into which we'll add the ground truth
                        # TODO: this is wasteful... label.render should probably
                        # accept the space object directly
                        truth = self.truth_space.create(array_creator=xp.zeros)
                        data = label.render(truth, xp=xp)
                        if self.settings.cache.write and cache_path:
                            to_cache(data, cache_path, dtype=np.uint16)

                    label_data.append(data)

                # concat along the F axis
                fluors = [lbl.fluorophore for lbl in self.sample.labels]
                truth = xr.concat(label_data, dim=pd.Index(fluors, name=Axis.F))
                truth.attrs.update(units="fluorophores", long_name="Ground Truth")
                self._ground_truth = truth

        logger.info(
            f"Ground truth generated in {time.perf_counter() - _t0:.2f} seconds"
        )
        return self._ground_truth

    def _channel_fluorescence(self) -> list[list[ChannelFluorescence]]:
        """Return how each fluorophore appears in each channel (`[channel][fluor]`).

        Each item describes one fluorophore in one channel, without spatial
        information: how many photons it absorbs (and at which wavelengths) at the
        reference illumination intensity, and what fraction of them are detected (and
        at which wavelengths).  It is excited by the light sources of all channels in
        the same track (see `tracks`).  Fluorophores are the unique fluorophores in
        `sample.labels`.
        """
        fluors = list({lbl.fluorophore: None for lbl in self.sample.labels})
        key = self._fluorescence_cache_key(fluors)
        if key is not None and (cached := _FLUORESCENCE_CACHE.get(key)) is not None:
            _FLUORESCENCE_CACHE.move_to_end(key)
            # use this simulation's own (equal) channel and fluorophore objects
            return [
                [
                    replace(fl, channel=oc, fluorophore=fluor)
                    for fl, fluor in zip(row, fluors, strict=True)
                ]
                for row, oc in zip(cached, self.channels, strict=True)
            ]

        fluorescence: list[list[ChannelFluorescence]] = [[] for _ in self.channels]
        for t_idx, track in enumerate(self._track_indices()):
            channels = [self.channels[c] for c in track]
            for fluor in fluors:
                # all light sources of the track excite the fluorophore
                absorption = track_absorption_rate(channels, fluor)
                for c_idx in track:
                    fl = ChannelFluorescence.create(
                        self.channels[c_idx],
                        fluor,
                        self.objective_lens,
                        self.detector,
                        absorption_rate=absorption,
                        track=t_idx,
                    )
                    fluorescence[c_idx].append(fl)
        if key is not None:
            _FLUORESCENCE_CACHE[key] = fluorescence
            if len(_FLUORESCENCE_CACHE) > _FLUORESCENCE_CACHE_SIZE:
                _FLUORESCENCE_CACHE.popitem(last=False)
        return fluorescence

    def filtered_emission_rates(self, *, saturate: bool = True) -> xr.DataArray:
        """Return the emission rates for each fluorophore in each channel.

        Returns a (C, F, W) array of detected emission rates for each fluorophore in
        each channel, as a function of wavelength.  The units are photons/s.  The range
        of wavelengths will encompass the union of all the fluorophores' emission
        spectra, and the rates will be zero where the fluorophore does not emit.

        These are the rates for a fluorophore at the reference illumination intensity
        (`OpticalConfig.power`): anywhere in a widefield field of view, or at the
        center of the excitation spot for point-scanning modalities (e.g. `Confocal`).
        If `saturate` is True (the default), excitation saturation at that intensity
        is applied.

        Examples
        --------
        >>> sim = Simulation(...)
        >>> rates = sim.filtered_emission_rates()
        >>> rates.isel(c=0).plot.line(x='w') # plot emission of all fluors in channel 0
        >>> plt.show()
        """
        nested_rates = [
            [fl.detected_spectrum(saturate=saturate) for fl in row]
            for row in self._channel_fluorescence()
        ]
        # combine xarray objects along the C and F axes, with outer join on W
        rates: xr.DataArray = xr.combine_nested(  # type: ignore [assignment]
            nested_rates,
            concat_dim=[Axis.C, Axis.F],
            combine_attrs="override",
            join="outer",
            fill_value=0,
        ).transpose(Axis.C, Axis.F, Axis.W)
        return rates

    def _fluorescence_cache_key(self, fluors: list[Fluorophore]) -> tuple | None:
        """Key identifying `_channel_fluorescence` output, or None if unhashable.

        This depends only on the optics and fluorophores (not on where the
        fluorophores are), so it can be reused across simulations.
        """
        try:
            return (
                tuple(oc.model_dump_json() for oc in self.channels),
                str(self._track_indices()),
                tuple(f.model_dump_json() for f in fluors),
                self.detector.model_dump_json(include={"qe"}) if self.detector else "",
                self.objective_lens.collection_efficiency,
            )
        except PydanticSerializationError:
            return None

    def emission_flux(self, *, saturate: bool = True) -> xr.DataArray:
        """Return the spatial emission in photons per second (after emission filters).

        This is the ground truth data multiplied by the filtered emission rates, *prior*
        to convolution with the PSF.

        This multiplies the per-fluorophore emission rates by the ground truth data to
        get the total emission flux for each voxel in the ground truth. The return
        array has dimensions (C, F, Z, Y, X).  The units are photons/s.

        Note, this integrates over all wavelengths. For finer control over the emission
        spectrum, you may wish to directly combine `filtered_emission_rates` with the
        ground truth data as needed.

        This function is *NOT* used by `run()`. It is provided as a convenience for
        debugging and analysis. See also `optical_image_per_fluor`, which is similar to
        this function, but returns the image *after* convolution with the PSF.  That
        function may return a different result than simply convolving this function's
        output with a single PSF, as it more accurately models multi-wavelength
        emission spectra (as superposition of multiple PSFs).
        """
        truth = self.ground_truth()
        if Axis.F not in truth.coords:
            # we have no fluorophores to calculate
            return truth

        # total photons/s emitted by each fluorophore in each channel
        em_rates = self.filtered_emission_rates(saturate=saturate).sum(Axis.W)
        em_rates = em_rates.copy(data=self._xp.asarray(em_rates.data))
        total_flux = em_rates * truth
        total_flux.attrs.update(units="photon/sec", long_name="Emission Flux")

        # (C, F, Z, Y, X)
        return total_flux

    def optical_image_per_fluor(self) -> xr.DataArray:
        """Return the optical image for each channel/fluorophore combination.

        This is the emission from each fluorophore in each channel, after filtering by
        the optical configuration and convolution with the PSF.

        The return array has dimensions (C, F, Z, Y, X).  The units are photons/s.
        """
        # (C, F, Z, Y, X)
        return self.modality.render(
            self.ground_truth(),  # (F, Z, Y, X)
            self._channel_fluorescence(),  # [C][F]
            objective_lens=self.objective_lens,
            settings=self.settings,
            xp=self._xp,
        )

    def optical_image(self) -> xr.DataArray:
        """Return the optical image as delivered to the detector.

        This is the same as `optical_image_per_fluor`, but sums the contributions of all
        fluorophores in each channel (which a detector would not know). The return
        array has dimensions (C, Z, Y, X).  The units are photons/s.
        """
        logger.info("Creating optical_image ...")
        oipf = self.optical_image_per_fluor()
        return oipf.sum(Axis.F)  # (C, Z, Y, X)

    def digital_image(
        self,
        optical_image: xr.DataArray | None = None,
        *,
        exposure_ms: float | None = None,  # defaults to model exposure time
        with_detector_noise: bool = True,
    ) -> xr.DataArray:
        """Return the digital image as captured by the detector.

        This down-scales the optical image to the output space, and simulates the
        detector response.  The return array has dimensions (C, Z, Y, X).  The units
        are gray values, based on the bit-depth of the detector.  If there is no
        detector or `with_detector_noise` is False, the units are simply photons.
        """
        _t0 = time.perf_counter()
        logger.info("Creating digital_image ...")

        if optical_image is None:
            optical_image = self.optical_image()
        image = optical_image  # (C, Z, Y, X)

        # rescale to output space
        # TODO: consider how we would integrate detector pixel size
        # rather than a user-specified output space
        if self.output_space is not None:
            logger.info(f"Rescaling to output space {self.output_space.shape}")
            # A camera pixel collects all light falling on its area, so sum over Y/X.
            # A point scanner records the signal at each scan position, so average.
            # Each Z plane is a separate exposure at one focal position (it does not
            # collect light from neighboring focal planes), so always average over Z.
            mean_axes = (
                [Axis.Z]
                if self.modality.detection_type.sums_over_pixel_area
                else [str(d) for d in image.dims]
            )
            image = self.output_space.rescale(image, mean_axes=mean_axes)

        # simulate detector
        # NOTE: exposure is a per-pixel integration time (camera exposure, or pixel
        # dwell time for point-scanning). It must NOT be divided by the number of
        # pixels, even for point-scanning modalities.
        if exposure_ms is None:
            _cfg_exposures = {ch: ch.exposure_ms for ch in self.channels}
            ch_exposures: float | xr.DataArray = xr.DataArray(
                [
                    _cfg_exposures.get(ch) or self.exposure_ms
                    for ch in image.coords[Axis.C].values
                ],
                dims=Axis.C,
                coords={Axis.C: image.coords[Axis.C]},
            )
        else:
            ch_exposures = exposure_ms

        if self.detector is not None and with_detector_noise:
            logger.info(f"Simulating {type(self.detector).__name__} detector ...")
            image = self.detector.render(image, exposure_ms=ch_exposures, xp=self._xp)
            image.attrs.update(units="gray values")
        else:
            image = image * (ch_exposures / 1000)
            image.attrs.update(units="photons")

        # (C, Z, Y, X)
        logger.info(
            f"Digital image generated in {time.perf_counter() - _t0:.2f} seconds"
        )
        return image

    def run(self) -> xr.DataArray:
        """Run the complete simulation and return the result.

        This will also write a file to disk if `output_path` is set.
        """
        self._write(image := self.digital_image())
        return image

    def run_batch(self, samples: Iterable[Sample]) -> Iterator[xr.DataArray]:
        """Run this simulation once per sample, yielding each result.

        Everything but the sample is shared, so PSFs and emission rates are computed
        once and reused (see `Settings.cache.in_mem_size`).  Each sample gets its own
        random seed (derived from `settings.random_seed`, so batches are
        reproducible), and if `output_path` is set, result `i` is written to
        `<stem>_<i>.<suffix>`.

        Examples
        --------
        >>> images = list(sim.run_batch(samples))
        >>> stacked = xr.concat(images, dim="s")  # if all samples share a shape
        """
        seeds = np.random.SeedSequence(self.settings.random_seed)
        for i, sample in enumerate(samples):
            seed = None
            if self.settings.random_seed is not None:
                seed = int(seeds.spawn(1)[0].generate_state(1)[0])
            update: dict[str, Any] = {
                "sample": sample,
                "settings": self.settings.model_copy(update={"random_seed": seed}),
            }
            if self.output_path is not None:
                stem = f"{self.output_path.stem}_{i}"
                update["output_path"] = self.output_path.with_stem(stem)
            yield self.model_copy(update=update).run()

    def _truth_cache_path(
        self,
        label: "FluorophoreDistribution",
        truth_space: "Space",
        seed: int | None,
    ) -> Path | None:
        if not (lbl_path := label.cache_path()):
            return None
        if callable(label.concentration):
            return None

        truth_cache = Path(microsim_cache("ground_truth"), *lbl_path)
        shape = f"shape{'_'.join(str(x) for x in truth_space.shape)}"
        scale = f"scale{'_'.join(str(x) for x in truth_space.scale)}"
        conc = f"conc{label.concentration}"
        truth_cache = truth_cache / shape / scale / conc
        if hasattr(label.distribution, "is_random") and label.distribution.is_random():
            truth_cache = truth_cache / f"seed{seed}"
        return truth_cache

    def _write(self, result: xr.DataArray) -> None:
        if not self.output_path:
            return
        if hasattr(result.data, "get"):
            result = result.copy(data=result.data.get(), deep=False)
        # coerce backend arrays (e.g. jax) to a contiguous numpy array; numcodecs
        # calls np.array(..., copy=False), which raises on numpy>=2 otherwise.
        result = result.copy(data=np.ascontiguousarray(result.data), deep=False)
        with suppress(Exception):
            try:
                sim_data = self.model_dump_json()
            except PydanticSerializationError:
                sim_data = self.model_dump_json(exclude={"sample"})
            result.attrs = {"microsim.Simulation": sim_data}
        result.coords[Axis.C] = [c.name for c in result.coords[Axis.C].values]
        if self.output_path.suffix == ".zarr":
            # consolidated metadata is not part of the zarr v3 spec (zarr warns);
            # skip it so reads/writes stay spec-compliant across zarr 2 and 3.
            result.to_zarr(self.output_path, mode="w", consolidated=False)
        elif self.output_path.suffix in (".nc",):
            result.to_netcdf(self.output_path)
        elif self.output_path.suffix in (".tif", ".tiff"):
            try:
                import tifffile as tf
            except ImportError as e:
                raise ImportError(
                    "Please `pip install microsim[io]` to write TIFF files."
                ) from e

            tf.imwrite(self.output_path, np.asanyarray(result))

    def plot(self, transpose: bool = False, legend: bool = True) -> None:
        """Plot a summary of fluorophores and filters/configs used in the simulation.

        Requires `matplotlib`.

        As the number of channels increases, the plot may become crowded.  In this case,
        consider setting `transpose=True` to plot channels on the x-axis and
        fluorophores on the y-axis.
        """
        plot_simulation_summary(self, transpose=transpose, legend=legend)


def plot_simulation_summary(
    sim: Simulation, transpose: bool = False, legend: bool = True
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Please `pip install microsim[view]` to use plotting/viewing functions."
        ) from e

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

    fluorescence = sim._channel_fluorescence()
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
        for fl in fluorescence[ch_idx]:
            name = fl.fluorophore.name
            tot = fl.total_absorption_rate(saturate=False)
            fl.absorption_rate.plot.line(
                ax=ab_ax[ch_idx], x=Axis.W, label=f"{name} ({tot:.2f} phot/s tot)"
            )
            em_rate = fl.fluorophore.emission_per_absorbed_photon
            em_rate = em_rate * fl.total_absorption_rate()
            em_rate.plot.line(
                ax=ab_ax[ch_idx],
                label=f"{name} emission (at peak)",
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
        qe: float | Spectrum = 1.0
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

            for fl in fluorescence[ch_idx]:
                final = fl.detected_spectrum().isel({Axis.F: 0, Axis.C: 0})
                final.plot.line(
                    ax=f_ax[ch_idx],
                    label=f"{fl.fluorophore.name} collection "
                    f"({final.sum():.2f} phot/s tot)",
                )

        if legend:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

    # make sure lims are at least 400-750
    a, b = fp_ax[0].get_xlim()
    fp_ax[0].set_xlim(min(a, 450), max(b, 750))  # shared x-axis
    plt.tight_layout()
    plt.show()
