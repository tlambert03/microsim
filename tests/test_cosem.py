import sys

import numpy as np
import pytest
import tensorstore as ts

from microsim.cosem import CosemDataset, CosemImage, CosemView, manage, organelles


def test_cosem_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = CosemDataset.fetch("jrc_hela-3")
    assert dataset in CosemDataset.all().values()
    assert dataset.name in CosemDataset.names()

    assert dataset.em_layers
    assert dataset.segmentation_layers
    assert dataset.prediction_layers
    assert dataset.analysis_layers
    assert dataset.views
    assert isinstance(dataset.thumbnail, np.ndarray)

    with monkeypatch.context() as m:
        m.setattr(
            CosemImage, "read", lambda *args, **kwargs: ts.array(np.array((1, 2)))
        )
        dataset.read("fibsem-uint16", level=-1)
        dataset.read(("fibsem-uint16", "er_seg"), level=-1)


def test_cosem_image() -> None:
    # note, this is also testing _get_similar ... since the "real" name is jrc_hela-3
    dataset = CosemDataset.fetch("jrc_hela_3")
    img = dataset.image(name="fibsem-uint16")
    assert isinstance(img, CosemImage)
    assert img.bucket_key == "jrc_hela-3/jrc_hela-3.n5/em/fibsem-uint16"

    assert isinstance(img.read(-1), ts.TensorStore)

    with pytest.raises(ValueError, match="Image not found"):
        img = dataset.image(name="not real")


def test_cosem_view() -> None:
    orgs = organelles()
    assert "Centrosome" in orgs
    first_view = orgs["Centrosome"][0]
    assert isinstance(first_view, CosemView)
    assert first_view.name == "Centrosome"

    assert first_view in CosemView.filter(dataset_name="jrc_hela-2")

    assert CosemDataset.fetch("jrc_hela-2").view("Centrosome") == first_view


def test_cosem_manage(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["", "--help"])
        with pytest.raises(SystemExit):
            manage.main()

    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["", "info", "jrc_hela-3"])
        manage.main()

    with monkeypatch.context() as m:
        m.setattr(CosemDataset, "show", lambda *args, **kwargs: None)
        m.setattr(sys, "argv", ["", "show", "jrc_hela-3", "fibsem-uint16"])
        manage.main()

    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["", "clear_cache"])
        manage.main()


def test_cosem_simulation():
    from microsim import schema as ms
    from microsim.schema.optical_config import lib

    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(
            shape=(128, 1024, 1024), scale=(0.032, 0.032, 0.032)
        ),
        output_space={"downscale": 4},
        sample=ms.Sample(
            labels=[
                ms.FluorophoreDistribution(
                    distribution=ms.Cosem(dataset="jrc_hela-3", image="er-mem_pred"),
                    fluorophore="EGFP",
                ),
                ms.FluorophoreDistribution(
                    distribution=ms.Cosem(dataset="jrc_hela-3", image="mito-mem_pred"),
                    fluorophore="mCherry",
                ),
            ]
        ),
        channels=[lib.FITC, lib.TRITC],
        modality=ms.Confocal(),
        detector=ms.CameraCCD(qe=0.82, read_noise=6),
        output_path="h2-cf.tif",
    )

    sim.run()
