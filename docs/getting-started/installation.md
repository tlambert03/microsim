---
icon: lucide/download
---

# Installation

microsim requires Python 3.12 or later.

## From PyPI

```bash
pip install "microsim[all]"
```

!!! warning "Pin a specific version"

    microsim changes quickly, and releases can break the API or change
    simulation results for the same parameters. For any project you want to
    reproduce later, pin the exact version:

    ```bash
    pip install "microsim[all]==0.x.y"
    ```

    You can check the installed version with `microsim.__version__`.

The `[all]` extra installs the plotting, file I/O, and data-download
dependencies used throughout these docs. The base package (`pip install
microsim`) can run every simulation, but cannot plot results, write TIFF files,
or fetch ground truth from OpenOrganelle.

??? info "What each extra installs"

    | Extra   | Adds                                           | Needed for                                                                 |
    | ------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
    | `view`  | matplotlib, [ndv](https://github.com/pyapp-kit/ndv) | [`ortho_plot`][microsim.util.ortho_plot], [`Simulation.plot`][microsim.Simulation.plot], interactive viewing |
    | `io`    | tifffile, imageio                              | writing `.tif` output                                                      |
    | `cosem` | boto3, s3fs, supabase                          | [`CosemLabel`](../guide/samples.md#openorganelle-cosem) ground truth        |
    | `jax`   | jax                                            | the `jax` backend                                                          |
    | `torch` | torch                                          | the `torch` backend                                                        |
    | `all`   | `view`, `io`, `cosem`, and rich                |                                                                            |

## Latest development version

```bash
pip install "microsim[all] @ git+https://github.com/tlambert03/microsim"
```

The development version changes often.

## GPU acceleration

Most of the computation time goes into PSF generation and convolution. These
can run on a GPU through [CuPy](https://docs.cupy.dev/en/stable/install.html)
or [JAX](https://docs.jax.dev/en/latest/installation.html). Neither is
installed by default; follow their installation instructions for your GPU and
CUDA version, then select the backend:

```py
sim = ms.Simulation(..., settings=ms.Settings(np_backend="cupy"))
```

See [Settings](../guide/settings.md#array-backends) for details. A `torch`
backend also exists, but is incomplete.

## Development install

microsim uses [uv](https://docs.astral.sh/uv/) for development:

```bash
git clone https://github.com/tlambert03/microsim
cd microsim
uv sync
uv run pytest
```

To build these docs locally:

```bash
uv run --group docs zensical serve
```
