import json
from collections.abc import Mapping
from functools import cache
from typing import Any, Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, field_validator, model_validator

from microsim._field_types import ExtCoeff, Nanometers, Nanoseconds

__all__ = ["get_fluorophore", "get_microscope", "FPbaseFluorophore", "FPbaseMicroscope"]


### Models ###

SpectrumType = Literal[
    "A_2P", "BM", "BP", "BS", "BX", "EM", "EX", "LP", "PD", "QE", "AB"
]


class Spectrum(BaseModel):
    subtype: SpectrumType
    data: list[tuple[float, float]] = Field(..., repr=False)


class Filter(BaseModel):
    name: str
    manufacturer: str
    bandcenter: float | None
    bandwidth: float | None
    edge: float | None


class FilterSpectrum(Spectrum):
    ownerFilter: Filter


class SpectrumOwner(BaseModel):
    name: str
    spectrum: Spectrum


class State(BaseModel):
    id: int
    exMax: Nanometers
    emMax: Nanometers
    extCoeff: ExtCoeff | None = None
    qy: float | None = None
    spectra: list[Spectrum]
    lifetime: Nanoseconds | None = None

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


class FilterPlacement(SpectrumOwner):
    path: Literal["EX", "EM", "BS"]
    reflects: bool = False


class OpticalConfig(BaseModel):
    name: str
    filters: list[FilterPlacement]
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


class FilterSpectrumPayload(BaseModel):
    spectrum: FilterSpectrum


class FilterSpectrumResponse(BaseModel):
    data: FilterSpectrumPayload


### Graphql Queries ###

FPBASE_URL = "https://www.fpbase.org/graphql/"


def _fpbase_query(query: str) -> bytes:
    headers = {"Content-Type": "application/json", "User-Agent": "microsim"}
    data = json.dumps({"query": query}).encode("utf-8")
    req = Request(FPBASE_URL, data=data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        return response.read()  # type: ignore


@cache
def get_microscope(id: str = "i6WL2W") -> FPbaseMicroscope:
    query = """
    {{
        microscope(id: "{id}") {{
            id
            name
            opticalConfigs {{
                name
                filters {{
                    name
                    path
                    reflects
                    spectrum {{ subtype data }}
                }}
                camera {{ name spectrum {{ subtype data }} }}
                light {{ name spectrum {{ subtype data }} }}
                laser
            }}
        }}
    }}
    """
    resp = _fpbase_query(query.format(id=id))
    return MicroscopeResponse.model_validate_json(resp).data.microscope


@cache
def fluorophore_ids() -> dict:
    resp = _fpbase_query("{ dyes { id name slug } proteins { id name slug } }")
    data: dict[str, list[dict[str, str]]] = json.loads(resp)["data"]
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
    resp = _fpbase_query(query.format(id=id))
    return DyeResponse.model_validate_json(resp).data.dye


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
                lifetime
                spectra {{ subtype data }}
            }}
            defaultState {{
                id
            }}
        }}
    }}
    """
    resp = _fpbase_query(query.format(id=id))
    return ProteinResponse.model_validate_json(resp).data.protein


def get_filter(name: str) -> FilterSpectrum:
    if (name := _norm_name(name)) not in (catalog := filter_spectrum_ids()):
        raise ValueError(f"Filter {name!r} not found")
    query = """
    {{
        spectrum(id:{id}) {{
            subtype
            data
            ownerFilter {{
                name
                manufacturer
                bandcenter
                bandwidth
                edge
            }}
        }}
    }}
    """
    resp = _fpbase_query(query.format(id=catalog[name]))
    return FilterSpectrumResponse.model_validate_json(resp).data.spectrum


@cache
def filter_spectrum_ids() -> Mapping[str, int]:
    resp = _fpbase_query('{ spectra(category:"F") { id owner { name } } }')
    data: dict = json.loads(resp)["data"]["spectra"]
    return {_norm_name(item["owner"]["name"]): int(item["id"]) for item in data}


def _norm_name(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")
