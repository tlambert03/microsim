from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class SimBaseModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        validate_default=True,
    )
