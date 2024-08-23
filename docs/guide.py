# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: microsim
#     language: python
#     name: python3
# ---

# # microsim guide
#
# This guide walks through some of the basic concepts of microsim, with examples
#
# ## Creating a sample
#
# The simulation begins with some sort of sample or "ground truth".  This is
# essentially a spatial distribution of signal-emitting objects.  `microsim`
# provides a few built-in sample distributions, but you can also create your own.
#
# The distribution is represented as a count of fluorophores in a 3-dimensional
# space.  That space is referred to as the *truth space*.  So, a very basic sample
# example, drawing some randomly oriented lines into a 64x256x256 space, with a
# ground truth voxel size of 10nm x 10nm x 10nm, might look like this:

# +
from microsim import schema as ms
from microsim.util import ortho_plot

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(64, 256, 256), scale=0.01),
    sample=[ms.MatsLines()],
)

ortho_plot(sim.ground_truth(), mip=True)
# -

# ## Biologically realistic ground truth from Janelia CellMap
#
# We can also create more biologically realistic ground truth.  For example,
# microsim provides a wrapper around the Janelia
# [OpenOrganelle](https://www.openorganelle.org) dataset, which is a rich dataset
# of FIB-SEM images of cells, together with annotations of various organelles.
# Let's simulate some ER, mitochondria, and lysosomes in a 3D space:

# +
from microsim import schema as ms

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(
        shape=(64, 512, 512),
        scale=(0.064, 0.064, 0.064),
    ),
    # dataset & label names from https://openorganelle.janelia.org/datasets
    sample=[
        ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
        ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
        ms.CosemLabel(dataset="jrc_hela-3", label="lyso_pred"),
    ],
)

ortho_plot(sim.ground_truth())
# -

# (or, plotted as maximum intensity projection):

ortho_plot(sim.ground_truth(), mip=True)

# Without any additional config, we can simulate an image as acquired by a
# widefield microscope:

# +
result = sim.digital_image()

ortho_plot(result)
# -

# interesting ... looks more or less like a widefield image of a cell.
# But there's something wrong: all of the organelles are in the same channel!
# Let's add the concept of fluorophores and optical configurations.

# ## Fluorophores, Spectra, and Optical Configurations

# Here we'll turn the three `CosemLabel` objects into `FluorophoreDistribution`
# objects, each with a specified `fluorophore` (the information for each fluorophore is
# pulled from [FPbase](https://fpbase.org).  We'll also specify the `channels`
# in the simulation, which is a list of `OpticalConfiguration` objects.  These objects
# describe the optical configuration of the microscope, including the excitation and
# emission spectra of the light sources and detectors (The
# `microsim.schema.optical_config.lib` module provides a library of common optical
# configurations)

# +
from microsim import schema as ms
from microsim.schema.optical_config import lib

sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(
        shape=(64, 512, 512),
        scale=(0.064, 0.064, 0.064),
    ),
    # dataset & label names from https://openorganelle.janelia.org/datasets
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
            fluorophore="EGFP",
        ),
        ms.FluorophoreDistribution(
            distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
            fluorophore="mCherry",
        ),
        ms.FluorophoreDistribution(
            distribution=ms.CosemLabel(dataset="jrc_hela-3", label="lyso_pred"),
            fluorophore="Alexa Fluor 647",
        ),
    ],
    channels=[lib.FITC, lib.TRITC, lib.CY7],
)
# -


# Let's first discuss how the fluorophores are excited and collected by our optical
# configurations. We can visualize the absorption spectra of the fluorophores in our
# sample:

# +
from matplotlib import pyplot as plt

for lbl in sim.sample.labels:
    f = lbl.fluorophore
    f.absorption_cross_section.plot.line(label=f.name)

plt.legend()
plt.show()
# -

# The excitation spectra have been scaled here by the extinction coefficient of the
# fluorophore and converted to
# [absorption cross-section](https://en.wikipedia.org/wiki/Absorption_cross_section)
# ($\sigma$) using the formula:

# $$
#   \sigma = \log(10) \times 10^3 \times \frac{E(\lambda)}{N_A}
# $$

# Where $E(\lambda)$ is the extinction coefficient in $M^{-1} cm^{-1}$, and $N_A$ is
# Avogadro's number.  The absorption cross-section is in $cm^2$.

# Each of the optical configurations determines how effectively the fluorophores are
# excited and collected

sim.plot()


for ch_idx, oc in enumerate(sim.channels):
    oc.plot_excitation()

    for f in oc.filters:
        if f.placement == Placement.EM_PATH:
            continue

        spect = f.spectrum
        if f.placement == Placement.BS:
            spect = spect.inverted()
        ex_ax2.plot(spect.wavelength, spect.intensity, label=f"{f.name}", alpha=0.4)
    # light sources
    for light in oc.lights:
        ls = light.spectrum
        ex_ax2.plot(ls.wavelength, ls.intensity, label=f"{light.name}", alpha=0.4)

    # combined illumination
    full = oc.illumination_flux_density
    full.plot.line(ax=ex_ax[ch_idx], label="flux density", color="k")
