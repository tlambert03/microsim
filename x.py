from matplotlib import pyplot as plt
from rich import print

from microsim.schema import OpticalConfig

oc = OpticalConfig.from_fpbase("i6WL2WdgcDMgJYtPrpZcaJ::Widefield Dual Green")
print(oc)

oc.plot()
