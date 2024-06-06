from typing import Annotated, Any

from annotated_types import Ge
from pydantic import BaseModel, model_validator

PositiveInteger = Annotated[int, Ge(0)]


class Person(BaseModel):
    name: str
    age: PositiveInteger

    @model_validator(mode="before")
    @classmethod
    def _validate_person(cls, value: Any) -> dict:
        if isinstance(value, str) and value in DATABASE:
            return DATABASE[value]
        return value


DATABASE = {
    "Bob": {"name": "Bob", "age": -25},
}


class Model(BaseModel):
    person: Person


b = Model(person="Bob")
