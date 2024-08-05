import microsim.schema as ms
import matplotlib.pyplot as plt

oc = ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green")
egfp = ms.Fluorophore.from_fpbase("EGFP")

r = oc.absorption_rate(egfp)
r.plot()
plt.show()