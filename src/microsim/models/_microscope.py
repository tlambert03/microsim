from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from microsim.illum import Widefield

from ._camera import ICX285, Camera
from ._coverslip import Coverslip
from ._illum import Illumination
from ._immersion import ImmersionMedium, OilImmersion
from ._objective import Objective


class Microscope(BaseModel):
    objective: Objective = Field(default_factory=Objective)
    coverslip: Coverslip = Field(default_factory=Coverslip)
    immersion: ImmersionMedium = OilImmersion()
    illumination: Illumination = Field(default_factory=Widefield)
    camera: Camera = ICX285

    class Config:
        validate_assignment = True

    @staticmethod
    def active() -> "Microscope":
        return _GLOBAL_MICROSCOPE.get()

    @staticmethod  # type: ignore
    @contextmanager
    def context(
        objective: Optional[Objective] = None,
        coverslip: Optional[Coverslip] = None,
        immersion: Optional[ImmersionMedium] = None,
        illumination: Optional[Illumination] = None,
        camera: Optional[Camera] = None,
    ) -> Iterable["Microscope"]:
        kwargs = {k: v for k, v in locals().items() if v is not None}
        current = _GLOBAL_MICROSCOPE.get().model_dump()

        token = _GLOBAL_MICROSCOPE.set(Microscope(**{**current, **kwargs}))
        try:
            yield _GLOBAL_MICROSCOPE.get()
        finally:
            _GLOBAL_MICROSCOPE.reset(token)


_GLOBAL_MICROSCOPE: ContextVar[Microscope] = ContextVar(
    "_GLOBAL_MICROSCOPE", default=Microscope()
)
