import pytest

from microsim.schema import Fluorophore, OpticalConfig, Simulation, SpectrumFilter
from tests._util import skipif_mpl_py314, skipif_no_internet


@skipif_mpl_py314
@skipif_no_internet
def test_plot_oc() -> None:
    pytest.importorskip("matplotlib")
    Fluorophore.from_fpbase("EGFP").plot(show=False)
    SpectrumFilter.from_fpbase("Chroma ET525/50m").plot()
    oc = OpticalConfig.from_fpbase("wKqWb::Widefield Green")
    oc.plot(show=False)
    oc.emission.plot()
    oc.excitation.plot()
    oc.plot_emission()
    oc.plot_excitation()


@skipif_mpl_py314
def test_plot_sim(sim1: Simulation) -> None:
    pytest.importorskip("matplotlib")
    sim1.plot()
