from microsim.schema import Fluorophore, OpticalConfig, SpectrumFilter


def test_plot() -> None:
    Fluorophore.from_fpbase("EGFP").plot(show=False)
    SpectrumFilter.from_fpbase("Chroma ET525/50m").plot(show=False)
    oc = OpticalConfig.from_fpbase("wKqWb::Widefield Green")
    oc.plot(show=False)
    oc.emission.plot(show=False)  # type: ignore
    oc.excitation.plot(show=False)  # type: ignore
