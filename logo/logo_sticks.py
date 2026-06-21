"""Prototype a minimal PLIMA logo with colored elliptical IA rings."""

from __future__ import annotations

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np


RNG_SEED = 7


def draw_stick(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    *,
    color: str,
    length: float = 0.28,
    linewidth: float = 3.0,
) -> None:
    """Draw a short stick centered at a position.

    Args:
        ax: Matplotlib axes to draw on.
        x: X coordinate of the stick center.
        y: Y coordinate of the stick center.
        theta: Orientation angle in radians.
        color: Stick color.
        length: Total stick length.
        linewidth: Stick line width.
    """
    dx = 0.5 * length * np.cos(theta)
    dy = 0.5 * length * np.sin(theta)

    ax.plot(
        [x - dx, x + dx],
        [y - dy, y + dy],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
    )


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
    n_sticks: int,
    colors: list[str],
    length: float,
    linewidth: float,
    ellipse_rotation: float = 0.0,
    phase: float = 0.0,
    angle_jitter_deg: float = 0.0,
    radial_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw a ring of mostly radially aligned sticks on an ellipse.

    Args:
        ax: Matplotlib axes to draw on.
        a: Ellipse semimajor axis.
        b: Ellipse semiminor axis.
        n_sticks: Number of sticks in the ring.
        colors: Colors to cycle through for the sticks.
        length: Stick length.
        linewidth: Stick line width.
        ellipse_rotation: Global ellipse rotation angle in radians.
        phase: Angular phase shift in radians.
        angle_jitter_deg: Gaussian orientation scatter in degrees.
        radial_jitter: Gaussian fractional scatter in ellipse radius.
        rng: Optional random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.linspace(0.0, 2.0 * np.pi, n_sticks, endpoint=False) + phase

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

        draw_stick(
            ax,
            x,
            y,
            theta,
            color=color,
            length=length,
            linewidth=linewidth,
        )


rng = np.random.default_rng(RNG_SEED)

# Take 5 colors from cmr.cosmic, clipping off the first 0.2 of the map.
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
    s=35,
    color=colors[0],
    zorder=3,
)

# Inner elliptical ring.
draw_elliptical_ia_ring(
    ax,
    a=0.68,
    b=0.48,
    n_sticks=8,
    colors=colors,
    length=0.20,
    linewidth=3.0,
    ellipse_rotation=np.deg2rad(25.0),
    phase=np.pi / 8.0,
    angle_jitter_deg=4.0,
    radial_jitter=0.015,
    rng=rng,
)

# Outer elliptical ring.
draw_elliptical_ia_ring(
    ax,
    a=1.18,
    b=0.88,
    n_sticks=10,
    colors=colors[::-1],
    length=0.24,
    linewidth=3.0,
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