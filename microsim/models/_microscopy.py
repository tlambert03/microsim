from pydantic import BaseModel

from ._coverslip import Coverslip
from ._objective import Objective


class Microscope(BaseModel):
    objective: Objective = Objective()
    coverslip: Coverslip = Coverslip()
