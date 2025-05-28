class Simulation:
    """Top level Simulation object."""

    truth_space: {"shape": ..., "scale": ...}
    output_space: {"shape": ..., "scale": ...}
    sample: list[FluorophoreDistribution]
    modality: Confocal | Widefield | SIM | TIRF | STED | SMLM | ...
    objective_lens: ObjectiveLens
    channels: list[OpticalConfig]


class FluorophoreDistribution:
    """Model of distribution and identity of fluorophores in space."""

    distribution: ThingThatCanRenderIntoSpace
    fluorophore: Fluorophore


class OpticalConfig:
    """Model of microscope optical configuration."""

    filters: list[Filter]
    lights: list[LightSource]
    detector: Detector
    exposure_ms: float


class Detector:
    """Model of a detector in a microscope."""

    camera_type: CCD | CMOS | EMCCD
    read_noise: float
    qe: float
    full_well_capacity: int
    dark_current: float
    bit_depth: int
    offset: int = 100
    gain: float
