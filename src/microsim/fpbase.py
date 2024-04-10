import json
from functools import cache
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

FPBASE_URL = "https://www.fpbase.org/graphql/"


class Spectrum(BaseModel):
    subtype: str
    data: list[tuple[float, float]] = Field(..., repr=False)


class SpectrumOwner(BaseModel):
    name: str
    spectrum: Spectrum


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


class GQLResponse(BaseModel):
    data: MicroscopePayload


@cache
def get_microscope(id: str) -> FPbaseMicroscope:
    query = """
    {{
        microscope(id: "{id}") {{
            id
            name
            opticalConfigs {{
                name
                filters {{
                    name
                    spectrum {{ subtype data }}
                }}
                camera {{
                    name
                    spectrum {{ subtype data }}
                }}
                light {{
                    name
                    spectrum {{ subtype data }}
                }}
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
        resp = GQLResponse.model_validate_json(response.read().decode("utf-8"))
        return resp.data.microscope


# ALL AVAILABLE SPECTRA
# {
# 	spectra{
#     id
#     subtype
#     category
#     owner{
#       name
#       id
#     }
#   }
# }

# ALL AVAILABLE DYES AND PROTEINS
# {
#   dyes {
#     id
#     name
#   }
#   proteins{
#     id
#     name
#   }
# }
