from microsim.illum._sim import SIMIllum3D
from microsim.util import uniformly_spaced_xarray

truth_shape = (256, 512, 512)
dz = 0.02
dx = 0.01

truth_space = uniformly_spaced_xarray(shape=truth_shape, scale=(dz, dx, dx))

# this API is going to change
# sample = MatsLines().render(truth_space)
illumination = SIMIllum3D().render(truth_space)
