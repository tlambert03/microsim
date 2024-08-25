import numpy as np
from scipy.optimize import fsolve


def pindist(t: float, t0: float, w: float, distance: float) -> float:
    # general function to minimize when finding the next "time" point to lay a
    # pinhole when traversing an archimedean spiral of width w, while looking
    # for a pinhole spacing of `distance`...
    # this is just the euclidean distance between the pinhole at time t and
    # time t0, minus the desired pinhole distance
    x = w * t * np.cos(t) - w * t0 * np.cos(t0)
    y = w * t * np.sin(t) - w * t0 * np.sin(t0)
    return x**2 + y**2 - distance**2  # type: ignore [no-any-return]


def pinhole_coords(
    radii: tuple[int, int] = (15, 25),
    pinhole_spacing: float = 0.253,
    frame_per_rev: float = 12,
    spiral_spacing: float | None = None,
) -> np.ndarray:
    # this works ... but is an ugly direct translation from MATLAB code.
    # TODO: get rid of forloops and actually use numpy.
    # defaults are for X1

    _spiral_spacing = (spiral_spacing or pinhole_spacing) * frame_per_rev / (2 * np.pi)

    minturn = radii[0] / _spiral_spacing
    maxturn = radii[1] / _spiral_spacing

    _t = [minturn]
    t0 = minturn
    dt = 2 * np.pi / minturn
    while t0 < maxturn:
        t0 = fsolve(pindist, t0 + dt, (t0, _spiral_spacing, pinhole_spacing)).item()
        _t.append(t0)
        dt = t0 - _t[-2]

    t = np.asarray(_t)
    p = np.array([_spiral_spacing * t * np.cos(t), _spiral_spacing * t * np.sin(t)])

    thetas = np.deg2rad(np.arange(0, 360, 360 / frame_per_rev))

    V = np.empty((2, len(thetas), p.shape[1]))
    for i, theta in enumerate(thetas):
        # rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        V[:, i] = R @ p

    return V.reshape(2, -1).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.draw import disk

    points = (pinhole_coords() * 100).astype(int) + 2500
    arr = np.zeros((5010, 5010))
    for p in points:
        arr[disk(p, 5)] = 1

    plt.imshow(arr)
    plt.show()
