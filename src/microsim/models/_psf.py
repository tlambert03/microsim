from pydantic import BaseModel

from ._coverslip import Coverslip
from ._immersion import ImmersionMedium
from ._objective import Objective


class PSF(BaseModel):
    objective: Objective
    coverslip: Coverslip
    immersion_medium: ImmersionMedium
    depth: float  # depth from the coverslip
