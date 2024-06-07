from microsim.allen import Specimen
from microsim.util import view_nd

spec = Specimen.fetch(555241040)
swc = spec.neuron_reconstructions[0].load_swc()
print(spec.id)

mask = swc.build_mask(global_scale_factor=2)
view_nd(mask)
