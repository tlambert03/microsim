from collections.abc import Sequence

import numpy as np
from psfmodels import library

from microsim.models import Coverslip, Objective

_DEFAULT_COVERSLIP = Coverslip()
_DEFAULT_OBJ = Objective.default()


# todo: pydantic validate arguments?
def vectorial_psf(
    objective: Objective = _DEFAULT_OBJ,
    zv=0,
    nx=31,
    dxy=0.05,
    particle_z=0.0,
    wavelength=0.6,  # in microns
    coverslip: Coverslip = _DEFAULT_COVERSLIP,
    immersion_ri=1.515,
    specimen_ri=1.515,
    normalize=True,
):
    if dxy <= 0:
        raise ValueError("dxy must be greater than 0")
    if particle_z < 0:
        raise ValueError("pz should be >= 0")

    _zv = _ensure_vector(zv).copy()
    _psf = library.vectorial_psf(
        _zv,
        int(nx),
        pz=particle_z,
        ti0=objective.working_distance,
        ni0=objective.immersion_ri,
        ni=immersion_ri,
        tg0=objective.cs_thickness_design,
        tg=coverslip.thickness,
        ng0=objective.cs_ri_design,
        ng=coverslip.ri,
        ns=specimen_ri,
        wvl=wavelength,
        NA=objective.na,
        dxy=dxy,
        sf=3,
        mode=1,
    )
    if normalize:
        _psf /= np.max(_psf)
    return _psf


def _ensure_vector(v: float | Sequence) -> np.ndarray:
    if np.isscalar(v):
        return np.array([v])
    elif isinstance(v, Sequence):
        return np.asarray(v)
    elif not isinstance(v, np.ndarray):
        raise ValueError("Value must be a scalar, iterable, or numpy array")
    return v
