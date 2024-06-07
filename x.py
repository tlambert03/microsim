from rich import print

from microsim.allen import get_reconstructions
from microsim.util import view_nd

for recon in get_reconstructions("Mus musculus", "full"):
    spec = recon.specimen()
    print("viewing", spec)
    view_nd(spec.binary_mask())
    break
