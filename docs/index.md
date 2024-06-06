# microsim

Microscope simulation library for generating realistic microscope images.

## Installation

For now, please install from github.  This library is in early development,
expect rapid changes and breakages.

```bash
pip install git+https://github.com/tlambert03/microsim
```

## Usage

Construct and run a [`microsim.Simulation`][] object.

=== "using objects"

    ```python
    from microsim import schema as ms
    from microsim.util import ortho_plot

    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.02, 0.01, 0.01)),
        output_space={'downscale': 8},
        sample=ms.Sample(
            labels=[ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1)]
        ),
        modality=ms.Confocal(pinhole_au=0.2),
    )
    result = sim.run()
    ortho_plot(result)
    ```

=== "using dicts"

    If you don't like importing all the names, you can also use
    dictionaries, and pydantic will convert them to the correct
    objects.
    
    ```python
    from microsim import Simulation
    from microsim.util import ortho_plot

    sim = Simulation(
        truth_space={'shape': (128, 512, 512), 'scale': (0.02, 0.01, 0.01)},
        output_space={'downscale': 8},
        sample=dict(
            labels=[
                {   
                    'type': 'matslines',
                    'density': 0.5,
                    'length': 30,
                    'azimuth': 5,
                    'max_r': 1,
                }
            ]
        ),
        modality={'type': 'confocal', 'pinhole_au': 0.2}
    )
    result = sim.run()
    ortho_plot(result)
    ```

![Image](./images/au02.png)

Note that you can also construct a simulation from a JSON object.
This is particularly useful with the `output` parameter, to dump
the result of the simulation to a file.

```json title="confocal.json"
{
    "truth_space": {
        "shape": [128, 512, 512],
        "scale": [0.02, 0.01, 0.01]
    },
    "output_space": {
        "downscale": 8
    },
    "sample": {
        "labels": [
        {
            "type": "matslines",
            "density": 0.5,
            "length": 30,
            "azimuth": 5,
            "max_r": 1
        }
        ]
    },
    "modality": {
        "type": "confocal",
        "pinhole_au": 0.2
    },
    "output": "au02.tiff"
}
```

... and then load a `Simulation` from that file and [`run()`][microsim.Simulation.run].

```python
from microsim import Simulation
from pathlib import Path

spec = Path('confocal.json').read_text()
sim = Simulation.model_validate_json(spec)
sim.run()
```
