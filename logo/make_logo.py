"""Generate and save the PLIMA elliptical intrinsic-alignment logo."""

from __future__ import annotations

from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


RNG_SEED = 7
OUTPUT_FILENAME = "logo-plima.png"


def find_repo_root(start: Path) -> Path:
    """Find the repository root containing pyproject.toml."""
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    msg = "Could not find the PLIMA repository root."
    raise FileNotFoundError(msg)


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
    """Draw a tiny elongated ellipse centered at a position."""
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
    """Rotate a point or array of points counterclockwise."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x_rotated = cos_angle * x - sin_angle * y
    y_rotated = sin_angle * x + cos_angle * y

    return x_rotated, y_rotated


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
    """Draw a ring of mostly radially aligned tiny ellipses."""
    if rng is None:
        rng = np.random.default_rng()

    angles = (
        np.linspace(
            0.0,
            2.0 * np.pi,
            n_ellipses,
            endpoint=False,
        )
        + phase
    )

    for index, phi in enumerate(angles):
        scale = 1.0 + rng.normal(
            loc=0.0,
            scale=radial_jitter,
        )

        x_unrotated = scale * a * np.cos(phi)
        y_unrotated = scale * b * np.sin(phi)

        x, y = rotate_xy(
            x_unrotated,
            y_unrotated,
            ellipse_rotation,
        )

        angle_jitter = np.deg2rad(
            rng.normal(
                loc=0.0,
                scale=angle_jitter_deg,
            )
        )

        theta = np.arctan2(y, x) + angle_jitter
        color = colors[index % len(colors)]

        draw_tiny_ellipse(
            ax,
            x,
            y,
            theta,
            color=color,
            length=length,
            width=width,
        )


def create_logo() -> tuple[plt.Figure, plt.Axes]:
    """Create the PLIMA logo figure."""
    rng = np.random.default_rng(RNG_SEED)

    colors = cmr.take_cmap_colors(
        "cmr.cosmic",
        5,
        cmap_range=(0.4, 1.0),
        return_fmt="hex",
    )

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        [0.0],
        [0.0],
        s=38,
        color=colors[0],
        zorder=3,
    )

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

    fig.tight_layout(pad=0.1)

    return fig, ax


def main() -> None:
    """Generate the logo and save it in the documentation assets directory."""
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    output_dir = repo_root / "docs" / "assets" / "logos"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILENAME

    fig, _ = create_logo()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(fig)

    print(f"Saved PLIMA logo to: {output_path}")


if __name__ == "__main__":
    main()