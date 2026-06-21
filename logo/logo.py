"""Prototype a minimal PLIMA logo with colored elliptical IA rings."""

from __future__ import annotations

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


RNG_SEED = 7


def draw_tiny_ellipse(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    *,
    color: str,
    length: float = 0.28,
    width: float = 0.055,
    alpha: float = 0.95,
    edgecolor: str | None = None,
    linewidth: float = 0.7,
) -> None:
    """Draw a tiny elongated ellipse centered at a position.

    Args:
        ax: Matplotlib axes to draw on.
        x: X coordinate of the ellipse center.
        y: Y coordinate of the ellipse center.
        theta: Orientation angle in radians.
        color: Ellipse face color.
        length: Ellipse major axis length.
        width: Ellipse minor axis width.
        alpha: Ellipse opacity.
        edgecolor: Optional ellipse edge color.
        linewidth: Ellipse edge line width.
    """
    ellipse = Ellipse(
        xy=(x, y),
        width=length,
        height=width,
        angle=np.rad2deg(theta),
        facecolor=color,
        edgecolor=edgecolor if edgecolor is not None else color,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(ellipse)


def rotate_xy(
    x: float | np.ndarray,
    y: float | np.ndarray,
    angle: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotate a point or array of points counterclockwise.

    Args:
        x: X coordinate or array of X coordinates.
        y: Y coordinate or array of Y coordinates.
        angle: Rotation angle in radians.

    Returns:
        Rotated X and Y coordinates.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y

    return x_rot, y_rot


def draw_elliptical_ia_ring(
    ax: plt.Axes,
    *,
    a: float,
    b: float,
    n_ellipses: int,
    colors: list[str],
    length: float,
    width: float,
    ellipse_rotation: float = 0.0,
    phase: float = 0.0,
    angle_jitter_deg: float = 0.0,
    radial_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw a ring of mostly radially aligned tiny ellipses.

    Args:
        ax: Matplotlib axes to draw on.
        a: Ellipse semimajor axis.
        b: Ellipse semiminor axis.
        n_ellipses: Number of tiny ellipses in the ring.
        colors: Colors to cycle through for the ellipses.
        length: Tiny ellipse major axis length.
        width: Tiny ellipse minor axis width.
        ellipse_rotation: Global ellipse rotation angle in radians.
        phase: Angular phase shift in radians.
        angle_jitter_deg: Gaussian orientation scatter in degrees.
        radial_jitter: Gaussian fractional scatter in ellipse radius.
        rng: Optional random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.linspace(0.0, 2.0 * np.pi, n_ellipses, endpoint=False) + phase

    for i, phi in enumerate(angles):
        scale = 1.0 + rng.normal(loc=0.0, scale=radial_jitter)

        x0 = scale * a * np.cos(phi)
        y0 = scale * b * np.sin(phi)

        x, y = rotate_xy(x0, y0, ellipse_rotation)

        angle_jitter = np.deg2rad(
            rng.normal(loc=0.0, scale=angle_jitter_deg)
        )

        theta = np.arctan2(y, x) + angle_jitter
        color = colors[i % len(colors)]

        draw_tiny_ellipse(
            ax,
            x,
            y,
            theta,
            color=color,
            length=length,
            width=width,
        )


rng = np.random.default_rng(RNG_SEED)

# Take 5 colors from cmr.cosmic, clipping off the first part of the map.
colors = cmr.take_cmap_colors(
    "cmr.cosmic",
    5,
    cmap_range=(0.4, 1.0),
    return_fmt="hex",
)

fig, ax = plt.subplots(figsize=(4, 4))

# Center density peak.
ax.scatter(
    [0.0],
    [0.0],
    s=38,
    color=colors[0],
    zorder=3,
)

# Inner elliptical IA ring.
# Inner elliptical IA ring: smaller galaxies.
draw_elliptical_ia_ring(
    ax,
    a=0.68,
    b=0.48,
    n_ellipses=8,
    colors=colors,
    length=0.17,
    width=0.040,
    ellipse_rotation=np.deg2rad(25.0),
    phase=np.pi / 8.0,
    angle_jitter_deg=4.0,
    radial_jitter=0.015,
    rng=rng,
)

# Outer elliptical IA ring: larger galaxies.
draw_elliptical_ia_ring(
    ax,
    a=1.18,
    b=0.88,
    n_ellipses=10,
    colors=colors[::-1],
    length=0.29,
    width=0.070,
    ellipse_rotation=np.deg2rad(25.0),
    phase=0.0,
    angle_jitter_deg=8.0,
    radial_jitter=0.020,
    rng=rng,
)

ax.set_aspect("equal")
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.55, 1.55)
ax.axis("off")

plt.tight_layout(pad=0.1)
plt.show()

# Save as vector if needed.
# fig.savefig("plima_logo_elliptical_ia_cosmic.svg", bbox_inches="tight", transparent=True)
# fig.savefig("plima_logo_elliptical_ia_cosmic.pdf", bbox_inches="tight", transparent=True)