import napari
import napari.types
from magicgui import magicgui

from microsim.illum._sim import structillum_2d

v = napari.Viewer()
kwargs = {
    "auto_call": True,
    "dz": {"max": 0.2, "step": 0.001},
    "dx": {"max": 0.2, "step": 0.001},
    "NA": {"max": 1.7, "min": 0.01, "step": 0.002},
    "nimm": {"max": 2, "min": 1, "step": 0.002},
    "wvl": {"min": 0.300, "max": 0.800, "step": 0.001},
    "linespacing": {"min": 0.05, "max": 0.8, "step": 0.001},
    "ampcenter": {"widget_type": "FloatSlider", "min": 0, "max": 2},
    "ampratio": {"widget_type": "FloatSlider", "min": 0, "max": 2},
    "nbeamlets": {"min": 1, "max": 101},
    "spotsize": {"max": 0.1, "min": 0.001, "step": 0.001},
}

structillum_2d.__annotations__["return"] = napari.types.ImageData
w = magicgui(structillum_2d, **kwargs)
v.window.add_dock_widget(w)
napari.run()
