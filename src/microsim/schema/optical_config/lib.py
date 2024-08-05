from microsim.schema.optical_config.config import OpticalConfig
from microsim.schema.optical_config.filter import Bandpass, Longpass

# https://www.chroma.com/products/sets/49000-et-dapi
DAPI = OpticalConfig(
    name="DAPI",
    filters=[
        Bandpass(bandcenter=350, bandwidth=50, placement="EX"),
        Longpass(cuton=400, placement="BS"),
        Bandpass(bandcenter=460, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49001-et-ecfp
ECFP = OpticalConfig(
    name="ECFP",
    filters=[
        Bandpass(bandcenter=436, bandwidth=20, placement="EX"),
        Longpass(cuton=455, placement="BS"),
        Bandpass(bandcenter=480, bandwidth=40, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49002-et-egfp-fitc-cy2
FITC = OpticalConfig(
    name="FITC",
    filters=[
        Bandpass(bandcenter=470, bandwidth=40, placement="EX"),
        Longpass(cuton=495, placement="BS"),
        Bandpass(bandcenter=525, bandwidth=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49003-et-eyfp
EYFP = OpticalConfig(
    name="EYFP",
    filters=[
        Bandpass(bandcenter=500, bandwidth=20, placement="EX"),
        Longpass(cuton=515, placement="BS"),
        Bandpass(bandcenter=535, bandwidth=30, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49004-et-cy3-tritc
TRITC = OpticalConfig(
    name="TRITC",
    filters=[
        Bandpass(bandcenter=545, bandwidth=25, placement="EX"),
        Longpass(cuton=565, placement="BS"),
        Bandpass(bandcenter=605, bandwidth=70, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49005-et-dsred-tritc-cy3
DSRED = OpticalConfig(
    name="DSRED",
    filters=[
        Bandpass(bandcenter=545, bandwidth=30, placement="EX"),
        Longpass(cuton=570, placement="BS"),
        Bandpass(bandcenter=620, bandwidth=60, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49006-et-cy5
CY5 = OpticalConfig(
    name="CY5",
    filters=[
        Bandpass(bandcenter=620, bandwidth=60, placement="EX"),
        Longpass(cuton=660, placement="BS"),
        Bandpass(bandcenter=700, bandwidth=75, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49007-et-cy7
CY7 = OpticalConfig(
    name="CY7",
    filters=[
        Bandpass(bandcenter=710, bandwidth=75, placement="EX"),
        Longpass(cuton=760, placement="BS"),
        Bandpass(bandcenter=810, bandwidth=90, placement="EM"),
    ],
)
