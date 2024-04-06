from pydantic import BaseModel


class ObjectiveLens(BaseModel):
    numerical_aperture: float = 1.4
