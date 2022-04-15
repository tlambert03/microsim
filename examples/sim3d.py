import napari

from microsim.models._illum import SIMIllum3D
from microsim.util import uniformly_spaced_xarray

space = uniformly_spaced_xarray(shape=(65, 128, 96), scale=(0.02, 0.01, 0.01))

illum3 = SIMIllum3D()


img = illum3.render(space)
viewer = napari.view_image(img)


napari.run()
