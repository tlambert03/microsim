import napari
from magicgui import magicgui

from microsim.models._illum import structillum_2d

v = napari.Viewer()
kwargs = dict(
    auto_call=True,
    dz=dict(max=0.2, step=0.001),
    dx=dict(max=0.2, step=0.001),
    NA=dict(max=1.7, min=0.01, step=0.002),
    nimm=dict(max=2, min=1, step=0.002),
    wvl=dict(min=0.300, max=0.800, step=0.001),
    linespacing=dict(min=0.05, max=0.8, step=0.001),
    ampcenter=dict(widget_type="FloatSlider", min=0, max=2),
    ampratio=dict(widget_type="FloatSlider", min=0, max=2),
    nbeamlets=dict(min=1, max=101),
    spotsize=dict(max=0.1, min=0.001, step=0.001),
)

w = magicgui(structillum_2d, **kwargs)
v.window.add_dock_widget(w)
napari.run()
