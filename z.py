import microsim.schema as ms

oc = ms.OpticalConfig.from_fpbase("wKqWbgApvguSNDSRZNSfpN", "Widefield Green")
egfp = ms.Fluorophore.from_fpbase("EGFP")

r = oc.absorption_rate(egfp)
print(r.sum())
