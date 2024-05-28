from microsim.schema import OpticalConfig
from microsim.schema.optical_config.filter import Bandpass

oc = OpticalConfig.from_fpbase("i6WL2WdgcDMgJYtPrpZcaJ::Widefield Dual Green")

oc.plot()
oc.excitation.plot()
oc.emission.plot()
