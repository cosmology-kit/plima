"""CCL backend helpers for LA intrinsic alignment bias inputs.

This module connects PLIMA LA amplitude models to the IA bias tuple expected by
CCL weak lensing tracers.

CCL weak lensing tracers expect the conventional user facing ``A_IA(z)``
amplitude when ``use_A_ia=True``. CCL then applies the IA normalization and
minus sign internally. Therefore this backend returns the PLIMA amplitude
directly and does not flip its sign.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from plima.models.la import la_amplitude
from plima.utils.types import FloatArray
from plima.utils.validators import as_finite_float_array, validate_greater_than

__all__ = [
    "make_ccl_la_ia_bias",
]


def make_ccl_la_ia_bias(
    z: ArrayLike,
    *,
    amplitude: ArrayLike | float | None = None,
    a_ia: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return a CCL IA bias tuple for LA.

    This backend follows the PLIMA user-facing convention that positive ``A_IA``
    corresponds to a positive LA amplitude. The returned ``ia_bias`` is the
    conventional ``A_IA(z)`` tuple expected by CCL when
    ``WeakLensingTracer(..., use_A_ia=True)`` is used.

    Args:
        z: Redshift values where the IA bias should be sampled.
        amplitude: Optional precomputed positive LA amplitude evaluated at ``z``.
            If ``None``, the amplitude is computed from ``la_amplitude``.
        a_ia: Positive LA amplitude normalization used when ``amplitude`` is
            ``None``.

    Returns:
        Redshift values and conventional CCL ``A_IA(z)`` values sampled on the
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
        physical_amplitude = la_amplitude(z_array, a_ia=a_ia)
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

    ia_bias = physical_amplitude

    return z_array.astype(np.float64, copy=True), ia_bias.astype(
        np.float64, copy=True
    )
