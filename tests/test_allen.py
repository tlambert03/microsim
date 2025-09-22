import numpy as np

from microsim.allen import NeuronReconstruction, Specimen

from ._util import skipif_no_internet


@skipif_no_internet
def test_neuron_reconstruction() -> None:
    nr = NeuronReconstruction.fetch(638976782)

    mask = nr.binary_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 3

    swc = nr.swc
    root = swc.root()
    assert root.id == 1
    assert swc.origin() == (root.z, root.y, root.x)


@skipif_no_internet
def test_specimen() -> None:
    spec = Specimen.fetch(586073850)
    masks = spec.binary_masks()
    assert len(masks) == len(spec.neuron_reconstructions)
    assert isinstance(spec.url, str)
