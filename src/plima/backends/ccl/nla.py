"""CCL backend helpers for NLA intrinsic alignment bias inputs.

This module connects PLIMA NLA amplitude models to the IA bias tuple expected
by CCL weak lensing tracers.

The backend independent NLA model uses a positive physical ``A_IA`` and applies
the physical NLA response sign internally. CCL weak lensing tracers use the
opposite sign convention for the IA bias tuple, so this backend flips the sign
before returning ``(z, ia_bias)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from plima.models.nla import nla_amplitude
from plima.utils.types import FloatArray
from plima.utils.validators import as_finite_float_array, validate_greater_than

__all__ = [
    "make_ccl_nla_ia_bias",
]


def make_ccl_nla_ia_bias(
    z: ArrayLike,
    *,
    amplitude: ArrayLike | float | None = None,
    a_ia: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return a CCL IA bias tuple for NLA.

    This backend follows the PLIMA user-facing convention that positive
    ``A_IA`` corresponds to a positive physical NLA amplitude. The sign needed
    by CCL weak lensing tracers is applied internally, so the returned
    ``ia_bias`` is ``-amplitude``.

    Args:
        z: Redshift values where the IA bias should be sampled.
        amplitude: Optional precomputed positive physical NLA amplitude
            evaluated at ``z``. If ``None``, the amplitude is computed from
            ``nla_amplitude``.
        a_ia: Positive physical NLA amplitude normalization used when
            ``amplitude`` is ``None``.

    Returns:
        Redshift values and CCL sign convention IA bias values sampled on the
        same grid.

    Raises:
        ValueError: If redshifts are not finite, redshifts are outside the
            valid domain, or amplitude cannot be broadcast to match ``z``.
    """
    z_array = np.atleast_1d(as_finite_float_array(z, name="z"))
    if z_array.size == 0:
        msg = "z must contain at least one value."
        raise ValueError(msg)

    validate_greater_than(z_array, threshold=-1.0, name="z")

    if amplitude is None:
        physical_amplitude = nla_amplitude(z_array, a_ia=a_ia)
    else:
        amplitude_array = as_finite_float_array(amplitude, name="amplitude")

        try:
            physical_amplitude = np.broadcast_to(
                amplitude_array,
                z_array.shape,
            ).astype(
                np.float64,
                copy=True,
            )
        except ValueError as error:
            msg = (
                "amplitude must be scalar or broadcastable to the same shape "
                "as z."
            )
            raise ValueError(msg) from error

    ia_bias = -physical_amplitude

    return z_array.astype(np.float64, copy=True), ia_bias.astype(
        np.float64, copy=True
    )
