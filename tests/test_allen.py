import numpy as np

from microsim.allen import NeuronReconstruction, Specimen


def test_neuron_reconstruction():
    nr = NeuronReconstruction.fetch(638976782)

    mask = nr.binary_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 3

    swc = nr.swc
    root = swc.root()
    assert root.id == 1
    assert swc.origin() == (root.z, root.y, root.x)


def test_specimen():
    spec = Specimen.fetch(586073850)
    masks = spec.binary_masks()
    assert len(masks) == len(spec.neuron_reconstructions)
    assert isinstance(spec.url, str)
