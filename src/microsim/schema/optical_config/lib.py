from microsim.schema.optical_config.config import OpticalConfig
from microsim.schema.optical_config.filter import Bandpass, Longpass

# https://www.chroma.com/products/sets/49000-et-dapi
DAPI = OpticalConfig(
    name="DAPI",
    filters=[
        Bandpass(bandcenter_nm=350, bandwidth_nm=50, placement="EX"),
        Longpass(cuton_nm=400, placement="BS"),
        Bandpass(bandcenter_nm=460, bandwidth_nm=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49001-et-ecfp
ECFP = OpticalConfig(
    name="ECFP",
    filters=[
        Bandpass(bandcenter_nm=436, bandwidth_nm=20, placement="EX"),
        Longpass(cuton_nm=455, placement="BS"),
        Bandpass(bandcenter_nm=480, bandwidth_nm=40, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49002-et-egfp-fitc-cy2
FITC = OpticalConfig(
    name="FITC",
    filters=[
        Bandpass(bandcenter_nm=470, bandwidth_nm=40, placement="EX"),
        Longpass(cuton_nm=495, placement="BS"),
        Bandpass(bandcenter_nm=525, bandwidth_nm=50, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49003-et-eyfp
EYFP = OpticalConfig(
    name="EYFP",
    filters=[
        Bandpass(bandcenter_nm=500, bandwidth_nm=20, placement="EX"),
        Longpass(cuton_nm=515, placement="BS"),
        Bandpass(bandcenter_nm=535, bandwidth_nm=30, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49004-et-cy3-tritc
TRITC = OpticalConfig(
    name="TRITC",
    filters=[
        Bandpass(bandcenter_nm=545, bandwidth_nm=25, placement="EX"),
        Longpass(cuton_nm=565, placement="BS"),
        Bandpass(bandcenter_nm=605, bandwidth_nm=70, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49005-et-dsred-tritc-cy3
DSRED = OpticalConfig(
    name="DSRED",
    filters=[
        Bandpass(bandcenter_nm=545, bandwidth_nm=30, placement="EX"),
        Longpass(cuton_nm=570, placement="BS"),
        Bandpass(bandcenter_nm=620, bandwidth_nm=60, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49006-et-cy5
CY5 = OpticalConfig(
    name="CY5",
    filters=[
        Bandpass(bandcenter_nm=620, bandwidth_nm=60, placement="EX"),
        Longpass(cuton_nm=660, placement="BS"),
        Bandpass(bandcenter_nm=700, bandwidth_nm=75, placement="EM"),
    ],
)

# https://www.chroma.com/products/sets/49007-et-cy7
CY7 = OpticalConfig(
    name="CY7",
    filters=[
        Bandpass(bandcenter_nm=710, bandwidth_nm=75, placement="EX"),
        Longpass(cuton_nm=760, placement="BS"),
        Bandpass(bandcenter_nm=810, bandwidth_nm=90, placement="EM"),
    ],
)
