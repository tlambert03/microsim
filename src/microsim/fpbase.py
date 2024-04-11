import json
from functools import cache
from typing import Any, Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = ["get_fluorophore", "get_microscope", "FPbaseFluorophore", "FPbaseMicroscope"]

FPBASE_URL = "https://www.fpbase.org/graphql/"
SpectrumType = Literal[
    "A_2P", "BM", "BP", "BS", "BX", "EM", "EX", "LP", "PD", "QE", "AB"
]

### Models ###


class Spectrum(BaseModel):
    subtype: SpectrumType
    data: list[tuple[float, float]] = Field(..., repr=False)


class SpectrumOwner(BaseModel):
    name: str
    spectrum: Spectrum


class State(BaseModel):
    id: int
    exMax: float
    emMax: float
    extCoeff: float
    qy: float
    spectra: list[Spectrum]

    @property
    def excitation_spectrum(self) -> Spectrum | None:
        spect = next((s for s in self.spectra if s.subtype == "EX"), None)
        if not spect:
            spect = next((s for s in self.spectra if s.subtype == "AB"), None)
        return spect

    @property
    def emission_spectrum(self) -> Spectrum | None:
        return next((s for s in self.spectra if s.subtype == "EM"), None)


class FPbaseFluorophore(BaseModel):
    name: str
    id: str
    states: list[State] = Field(default_factory=list)
    defaultState: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _v_model(cls, v: Any) -> Any:
        if isinstance(v, dict):
            out = dict(v)
            if "states" not in v and "exMax" in v:
                out["states"] = [State(**v)]
            return out
        return v

    @field_validator("defaultState", mode="before")
    @classmethod
    def _v_default_state(cls, v: Any) -> int:
        if isinstance(v, dict) and "id" in v:
            return int(v["id"])
        return int(v)

    @property
    def default_state(self) -> State | None:
        for state in self.states:
            if state.id == self.defaultState:
                return state
        return next(iter(self.states), None)


class OpticalConfig(BaseModel):
    name: str
    filters: list[SpectrumOwner]
    camera: SpectrumOwner | None
    light: SpectrumOwner | None
    laser: int | None


class FPbaseMicroscope(BaseModel):
    id: str
    name: str
    opticalConfigs: list[OpticalConfig]


class MicroscopePayload(BaseModel):
    microscope: FPbaseMicroscope


class MicroscopeResponse(BaseModel):
    data: MicroscopePayload


class ProteinPayload(BaseModel):
    protein: FPbaseFluorophore


class ProteinResponse(BaseModel):
    data: ProteinPayload


class DyePayload(BaseModel):
    dye: FPbaseFluorophore


class DyeResponse(BaseModel):
    data: DyePayload


### Getter Functions ###


@cache
def get_microscope(id: str) -> FPbaseMicroscope:
    query = """
    {{
        microscope(id: "{id}") {{
            id
            name
            opticalConfigs {{
                name
                filters {{ name spectrum {{ subtype data }} }}
                camera {{ name spectrum {{ subtype data }} }}
                light {{ name spectrum {{ subtype data }} }}
                laser
            }}
        }}
    }}
    """
    headers = {"Content-Type": "application/json", "User-Agent": "microsim"}
    data = json.dumps({"query": query.format(id=id)}).encode("utf-8")
    req = Request(FPBASE_URL, data=data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        resp = MicroscopeResponse.model_validate_json(response.read())
        return resp.data.microscope


@cache
def fluorophore_ids() -> dict:
    query = "{ dyes { id name slug } proteins { id name slug } }"
    headers = {"Content-Type": "application/json", "User-Agent": "microsim"}
    query_data = json.dumps({"query": query}).encode("utf-8")
    req = Request(FPBASE_URL, data=query_data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        data: dict[str, list[dict[str, str]]] = json.load(response)["data"]

    lookup: dict[str, dict[str, str]] = {}
    for key in ["dyes", "proteins"]:
        for item in data[key]:
            lookup[item["name"].lower()] = {"id": item["id"], "type": key[0]}
            lookup[item["slug"]] = {"id": item["id"], "type": key[0]}
            if key == "proteins":
                lookup[item["id"]] = {"id": item["id"], "type": key[0]}
    return lookup


@cache
def get_fluorophore(id: str) -> FPbaseFluorophore:
    try:
        fluor_info = fluorophore_ids()[id.lower()]
    except KeyError as e:
        raise ValueError(f"Fluorophore {id!r} not found") from e

    if fluor_info["type"] == "d":
        return get_dye_by_id(fluor_info["id"])
    elif fluor_info["type"] == "p":
        return get_protein_by_id(fluor_info["id"])
    raise ValueError(f"Invalid fluorophore type {fluor_info['type']!r}")


@cache
def get_dye_by_id(id: str | int) -> FPbaseFluorophore:
    query = """
    {{
        dye(id: {id}) {{
            name
            id
            exMax
            emMax
            extCoeff
            qy
            spectra {{ subtype data }}
        }}
    }}
    """
    headers = {"Content-Type": "application/json", "User-Agent": "microsim"}
    data = json.dumps({"query": query.format(id=id)}).encode("utf-8")
    req = Request(FPBASE_URL, data=data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        resp = DyeResponse.model_validate_json(response.read())
    return resp.data.dye


@cache
def get_protein_by_id(id: str) -> FPbaseFluorophore:
    query = """
    {{
        protein(id: "{id}") {{
            name
            id
            states {{
                id
                name
                exMax
                emMax
                extCoeff
                qy
                spectra {{ subtype data }}
            }}
            defaultState {{
                id
            }}
        }}
    }}
    """
    headers = {"Content-Type": "application/json", "User-Agent": "microsim"}
    data = json.dumps({"query": query.format(id=id)}).encode("utf-8")
    req = Request(FPBASE_URL, data=data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        resp = ProteinResponse.model_validate_json(response.read())
    return resp.data.protein
