from microsim.schema import Fluorophore, OpticalConfig, Simulation, SpectrumFilter


def test_plot_oc() -> None:
    Fluorophore.from_fpbase("EGFP").plot(show=False)
    SpectrumFilter.from_fpbase("Chroma ET525/50m").plot()
    oc = OpticalConfig.from_fpbase("wKqWb::Widefield Green")
    oc.plot(show=False)
    oc.emission.plot()
    oc.excitation.plot()
    oc.plot_emission()
    oc.plot_excitation()


def test_plot_sim(sim1: Simulation) -> None:
    sim1.plot()
