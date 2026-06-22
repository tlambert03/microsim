"""Visualize the axial PSF profile to show optical sectioning."""

import matplotlib.pyplot as plt
import numpy as np

from microsim import schema as ms
from microsim.schema.optical_config import lib

# Parameters from x.py
pinhole_au = 2.0
channel = lib.CY5

ex_wvl_nm = channel.filters[0].bandcenter
em_wvl_nm = channel.filters[2].bandcenter
objective = ms.ObjectiveLens()

# Create PSF
test_z_extent = 4.0  # μm
dz = 0.032
nz = int(test_z_extent / dz)
if nz % 2 == 0:
    nz += 1

dx = 0.032
nx = 65

from microsim.psf import make_confocal_psf

psf = make_confocal_psf(
    nz=nz,
    ex_wvl_um=ex_wvl_nm / 1000,
    em_wvl_um=em_wvl_nm / 1000,
    pinhole_au=pinhole_au,
    nx=nx,
    ny=nx,
    dz=dz,
    dxy=dx,
    objective=objective,
)

# Extract axial profile
center_xy = nx // 2
axial_profile = psf[:, center_xy, center_xy]
axial_profile = axial_profile / axial_profile.max()

# Create z coordinates centered at 0
z_coords = (np.arange(nz) - nz // 2) * dz

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


# Calculate distances at different thresholds
def find_extent_at_threshold(profile, z_coords, threshold):
    """Find the half-extent (radius) where profile drops to threshold."""
    above = np.where(profile >= threshold)[0]
    if len(above) > 0:
        return abs(z_coords[above[-1]])  # distance from center to edge
    return 0


extent_50 = find_extent_at_threshold(axial_profile, z_coords, 0.5)
extent_10 = find_extent_at_threshold(axial_profile, z_coords, 0.1)
extent_01 = find_extent_at_threshold(axial_profile, z_coords, 0.01)

# Plot 1: Linear scale
ax1.plot(z_coords, axial_profile, "b-", linewidth=2, label="Confocal PSF")
ax1.axhline(
    y=0.5,
    color="r",
    linestyle="--",
    alpha=0.5,
    label=f"FWHM (50%) at ±{extent_50:.2f} μm",
)
ax1.axhline(
    y=0.1,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"10% at ±{extent_10:.2f} μm",
)
ax1.axhline(
    y=0.01,
    color="yellow",
    linestyle="--",
    alpha=0.5,
    label=f"1% at ±{extent_01:.2f} μm",
)

# Mark the current z extent from x.py
from x import sim

current_z_extent = sim.truth_space.shape[0] * sim.truth_space.scale[0]
nz_current = sim.truth_space.shape[0]
dz_current = sim.truth_space.scale[0]
ax1.axvspan(
    -current_z_extent / 2,
    current_z_extent / 2,
    alpha=0.2,
    color="green",
    label=f"Current truth space\n({nz_current} × {dz_current} = {current_z_extent:.3f} μm)",
)

# Mark recommended extents (distance from center to boundary on EACH SIDE)
fwhm_pixels = len(np.where(axial_profile >= 0.5)[0])
measured_fwhm = fwhm_pixels * dz

# Moderate: ~2× FWHM on each side (total = 4× FWHM)
recommended_moderate_radius = 2 * measured_fwhm
recommended_moderate_total = 2 * recommended_moderate_radius

# Conservative: ~3× FWHM on each side (total = 6× FWHM)
recommended_conservative_radius = 3 * measured_fwhm
recommended_conservative_total = 2 * recommended_conservative_radius

# Plot moderate recommendation
ax1.axvline(
    x=-recommended_moderate_radius,
    color="purple",
    linestyle=":",
    alpha=0.7,
    linewidth=2.5,
)
ax1.axvline(
    x=recommended_moderate_radius,
    color="purple",
    linestyle=":",
    alpha=0.7,
    linewidth=2.5,
    label=f"Moderate: 2×FWHM each side\n(Total: {recommended_moderate_total:.2f} μm)",
)

# Plot conservative recommendation
ax1.axvline(
    x=-recommended_conservative_radius,
    color="darkgreen",
    linestyle="-.",
    alpha=0.7,
    linewidth=2,
)
ax1.axvline(
    x=recommended_conservative_radius,
    color="darkgreen",
    linestyle="-.",
    alpha=0.7,
    linewidth=2,
    label=f"Conservative: 3×FWHM each side\n(Total: {recommended_conservative_total:.2f} μm)",
)

ax1.set_xlabel("Distance from focal plane (μm)", fontsize=12)
ax1.set_ylabel("Normalized PSF intensity", fontsize=12)
ax1.set_title(
    f"Confocal PSF Axial Profile (FWHM = {measured_fwhm:.2f} μm)\n(NA={objective.numerical_aperture}, λ={em_wvl_nm}nm, pinhole={pinhole_au}AU)",
    fontsize=13,
)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim([0, 1.05])
ax1.set_xlim([-2.5, 2.5])

# Calculate 0.1% extent too
extent_001 = find_extent_at_threshold(axial_profile, z_coords, 0.001)

# Plot 2: Log scale to see wings
ax2.semilogy(z_coords, axial_profile, "b-", linewidth=2)
ax2.axhline(
    y=0.5, color="r", linestyle="--", alpha=0.5, label=f"50% at ±{extent_50:.2f} μm"
)
ax2.axhline(
    y=0.1,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"10% at ±{extent_10:.2f} μm",
)
ax2.axhline(
    y=0.01,
    color="yellow",
    linestyle="--",
    alpha=0.5,
    label=f"1% at ±{extent_01:.2f} μm",
)
ax2.axhline(
    y=0.001,
    color="brown",
    linestyle="--",
    alpha=0.5,
    label=f"0.1% at ±{extent_001:.2f} μm",
)

ax2.axvspan(-current_z_extent / 2, current_z_extent / 2, alpha=0.2, color="green")
ax2.axvline(
    x=-recommended_moderate_radius,
    color="purple",
    linestyle=":",
    alpha=0.7,
    linewidth=2.5,
)
ax2.axvline(
    x=recommended_moderate_radius,
    color="purple",
    linestyle=":",
    alpha=0.7,
    linewidth=2.5,
)
ax2.axvline(
    x=-recommended_conservative_radius,
    color="darkgreen",
    linestyle="-.",
    alpha=0.7,
    linewidth=2,
)
ax2.axvline(
    x=recommended_conservative_radius,
    color="darkgreen",
    linestyle="-.",
    alpha=0.7,
    linewidth=2,
)

ax2.set_xlabel("Distance from focal plane (μm)", fontsize=12)
ax2.set_ylabel("Normalized PSF intensity (log scale)", fontsize=12)
ax2.set_title("PSF Profile (Log Scale)\nShowing extended wings", fontsize=13)
ax2.grid(True, alpha=0.3, which="both")
ax2.legend(loc="upper right", fontsize=8)
ax2.set_ylim([0.0001, 2])
ax2.set_xlim([-2.5, 2.5])

plt.tight_layout()
plt.savefig("psf_axial_profile.png", dpi=150, bbox_inches="tight")
print("✓ Saved visualization to psf_axial_profile.png")

# Also create a 2D cross-section visualization
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create YZ cross-section through center
yz_section = psf[:, :, center_xy]
extent = [-nx // 2 * dx, nx // 2 * dx, -nz // 2 * dz, nz // 2 * dz]

im = ax.imshow(
    yz_section,
    extent=extent,
    aspect="auto",
    cmap="hot",
    origin="lower",
    interpolation="bilinear",
)
ax.set_xlabel("Y position (μm)", fontsize=12)
ax.set_ylabel("Z position (μm)", fontsize=12)
ax.set_title(
    "Confocal PSF Cross-Section (YZ plane)\nShowing optical sectioning capability",
    fontsize=13,
)

# Mark the current truth space extent
ax.axhline(
    y=-current_z_extent / 2, color="cyan", linestyle="--", linewidth=2, alpha=0.8
)
ax.axhline(
    y=current_z_extent / 2,
    color="cyan",
    linestyle="--",
    linewidth=2,
    alpha=0.8,
    label=f"Current truth space\n({nz_current} × {dz_current} = {current_z_extent:.3f} μm)",
)

# Mark recommended extents
ax.axhline(
    y=-recommended_moderate_radius,
    color="lime",
    linestyle=":",
    linewidth=2.5,
    alpha=0.8,
)
ax.axhline(
    y=recommended_moderate_radius,
    color="lime",
    linestyle=":",
    linewidth=2.5,
    alpha=0.8,
    label=f"Moderate (2×FWHM each side)\nTotal: {recommended_moderate_total:.2f} μm",
)

ax.axhline(
    y=-recommended_conservative_radius,
    color="yellow",
    linestyle="-.",
    linewidth=2,
    alpha=0.8,
)
ax.axhline(
    y=recommended_conservative_radius,
    color="yellow",
    linestyle="-.",
    linewidth=2,
    alpha=0.8,
    label=f"Conservative (3×FWHM each side)\nTotal: {recommended_conservative_total:.2f} μm",
)

plt.colorbar(im, ax=ax, label="Normalized intensity")
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("psf_yz_cross_section.png", dpi=150, bbox_inches="tight")
print("✓ Saved cross-section to psf_yz_cross_section.png")

print("\nVisualization complete!")
