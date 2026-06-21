"""TATT intrinsic alignment amplitude models.

This module provides array safe helpers for tidal alignment and tidal torquing
amplitudes.

The functions here do not compute IA power spectra. They only prepare the
redshift dependent amplitudes and normalized coefficients needed by downstream
codes such as CCL or a Fisher forecasting pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from plima.models.model_registry import register_ia_model
from plima.utils.constants import C1_RHO_CRITICAL, DEFAULT_PIVOT_REDSHIFT
from plima.utils.types import FloatArray
from plima.utils.validators import (
    as_finite_float_array,
    validate_finite,
    validate_greater_than,
    validate_positive,
)

__all__ = [
    "tatt_amplitudes",
    "tatt_normalized_coefficients",
    "tatt_pt_biases",
    "unity_ia_bias",
]


@register_ia_model(
    "tatt",
    aliases=(
        "tatt_amplitudes",
        "tatt_a",
    ),
)
def tatt_amplitudes(
    z: ArrayLike,
    *,
    a1: float = 1.0,
    a2: float = 0.0,
    a1delta: float = 0.0,
    eta1: float = 0.0,
    eta2: float = 0.0,
    eta1delta: float = 0.0,
    z_pivot: float = DEFAULT_PIVOT_REDSHIFT,
) -> dict[str, FloatArray]:
    """Return redshift dependent TATT amplitudes.

    Args:
        z: Redshift values.
        a1: Linear tidal alignment amplitude at the pivot redshift.
        a2: Quadratic tidal torquing amplitude at the pivot redshift.
        a1delta: Source density weighting amplitude at the pivot redshift.
        eta1: Redshift evolution index for ``a1``.
        eta2: Redshift evolution index for ``a2``.
        eta1delta: Redshift evolution index for ``a1delta``.
        z_pivot: Pivot redshift.

    Returns:
        Dictionary containing ``a1``, ``a2``, and ``a1delta`` arrays.

    Raises:
        ValueError: If any numerical input is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If ``z_pivot`` is less than or equal to ``-1``.
    """
    z = as_finite_float_array(z, name="z")

    validate_finite(a1, name="a1")
    validate_finite(a2, name="a2")
    validate_finite(a1delta, name="a1delta")
    validate_finite(eta1, name="eta1")
    validate_finite(eta2, name="eta2")
    validate_finite(eta1delta, name="eta1delta")
    validate_finite(z_pivot, name="z_pivot")
    validate_greater_than(z, threshold=-1.0, name="z")
    validate_greater_than(z_pivot, threshold=-1.0, name="z_pivot")

    redshift_ratio = (1.0 + z) / (1.0 + z_pivot)

    return {
        "a1": np.asarray(a1 * redshift_ratio**eta1, dtype=np.float64),
        "a2": np.asarray(a2 * redshift_ratio**eta2, dtype=np.float64),
        "a1delta": np.asarray(
            a1delta * redshift_ratio**eta1delta,
            dtype=np.float64,
        ),
    }


def tatt_normalized_coefficients(
    z: ArrayLike,
    *,
    growth_factor: ArrayLike,
    omega_m: float,
    a1: float = 1.0,
    a2: float = 0.0,
    a1delta: float = 0.0,
    eta1: float = 0.0,
    eta2: float = 0.0,
    eta1delta: float = 0.0,
    z_pivot: float = DEFAULT_PIVOT_REDSHIFT,
    c1_rho_critical: float = C1_RHO_CRITICAL,
    omega_m_fid: float = 0.3,
    use_omega_m_squared_for_c2: bool = False,
) -> dict[str, FloatArray]:
    """Return normalized TATT coefficients.

    These are the coefficients used by PT IA tracers.

    ``c1`` is the linear tidal alignment coefficient.

    ``c2`` is the quadratic tidal torquing coefficient.

    ``cdelta`` is the source density weighting coefficient.

    Args:
        z: Redshift values.
        growth_factor: Linear growth factor evaluated at ``z``.
        omega_m: Matter density parameter for the cosmology.
        a1: Linear tidal alignment amplitude at the pivot redshift.
        a2: Quadratic tidal torquing amplitude at the pivot redshift.
        a1delta: Source density weighting amplitude at the pivot redshift.
        eta1: Redshift evolution index for ``a1``.
        eta2: Redshift evolution index for ``a2``.
        eta1delta: Redshift evolution index for ``a1delta``.
        z_pivot: Pivot redshift.
        c1_rho_critical: IA normalization constant.
        omega_m_fid: Fiducial matter density parameter used by one ``c2``
            convention.
        use_omega_m_squared_for_c2: If true, use the convention with
            ``omega_m ** 2 / omega_m_fid`` for ``c2``. If false, use the
            convention with one power of ``omega_m``.

    Returns:
        Dictionary containing ``c1``, ``c2``, and ``cdelta`` arrays.

    Raises:
        ValueError: If any numerical input is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If any growth factor value is not positive.
        ValueError: If ``omega_m`` is not positive.
        ValueError: If ``omega_m_fid`` is not positive.
    """
    z = as_finite_float_array(z, name="z")
    growth_factor = as_finite_float_array(
        growth_factor,
        name="growth_factor",
    )
    if growth_factor.shape != z.shape:
        msg = "growth_factor must have the same shape as z."
        raise ValueError(msg)

    validate_greater_than(z, threshold=-1.0, name="z")
    validate_positive(growth_factor, name="growth_factor")
    validate_positive(omega_m, name="omega_m")
    validate_positive(omega_m_fid, name="omega_m_fid")
    validate_finite(c1_rho_critical, name="c1_rho_critical")

    amplitudes = tatt_amplitudes(
        z,
        a1=a1,
        a2=a2,
        a1delta=a1delta,
        eta1=eta1,
        eta2=eta2,
        eta1delta=eta1delta,
        z_pivot=z_pivot,
    )

    a1_z = amplitudes["a1"]
    a2_z = amplitudes["a2"]
    a1delta_z = amplitudes["a1delta"]

    c1 = -a1_z * c1_rho_critical * omega_m / growth_factor
    cdelta = -a1delta_z * c1_rho_critical * omega_m / growth_factor

    if use_omega_m_squared_for_c2:
        c2_prefactor = 5.0 * c1_rho_critical * omega_m**2 / omega_m_fid
    else:
        c2_prefactor = 5.0 * c1_rho_critical * omega_m

    c2 = a2_z * c2_prefactor / growth_factor**2

    return {
        "c1": np.asarray(c1, dtype=np.float64),
        "c2": np.asarray(c2, dtype=np.float64),
        "cdelta": np.asarray(cdelta, dtype=np.float64),
    }


def tatt_pt_biases(
    z: ArrayLike,
    *,
    growth_factor: ArrayLike,
    omega_m: float,
    a1: float = 1.0,
    a2: float = 0.0,
    a1delta: float = 0.0,
    eta1: float = 0.0,
    eta2: float = 0.0,
    eta1delta: float = 0.0,
    z_pivot: float = DEFAULT_PIVOT_REDSHIFT,
    c1_rho_critical: float = C1_RHO_CRITICAL,
    omega_m_fid: float = 0.3,
    use_omega_m_squared_for_c2: bool = False,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Return PT bias inputs for a TATT IA tracer.

    Args:
        z: Redshift values.
        growth_factor: Linear growth factor evaluated at ``z``.
        omega_m: Matter density parameter for the cosmology.
        a1: Linear tidal alignment amplitude at the pivot redshift.
        a2: Quadratic tidal torquing amplitude at the pivot redshift.
        a1delta: Source density weighting amplitude at the pivot redshift.
        eta1: Redshift evolution index for ``a1``.
        eta2: Redshift evolution index for ``a2``.
        eta1delta: Redshift evolution index for ``a1delta``.
        z_pivot: Pivot redshift.
        c1_rho_critical: IA normalization constant.
        omega_m_fid: Fiducial matter density parameter used by one ``c2``
            convention.
        use_omega_m_squared_for_c2: If true, use the convention with
            ``omega_m ** 2 / omega_m_fid`` for ``c2``.

    Returns:
        Dictionary with ``c1``, ``c2``, and ``cdelta`` entries. Each entry is a
        ``(z, coefficient)`` tuple.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")

    coefficients = tatt_normalized_coefficients(
        z,
        growth_factor=growth_factor,
        omega_m=omega_m,
        a1=a1,
        a2=a2,
        a1delta=a1delta,
        eta1=eta1,
        eta2=eta2,
        eta1delta=eta1delta,
        z_pivot=z_pivot,
        c1_rho_critical=c1_rho_critical,
        omega_m_fid=omega_m_fid,
        use_omega_m_squared_for_c2=use_omega_m_squared_for_c2,
    )

    return {
        "c1": (z, coefficients["c1"]),
        "c2": (z, coefficients["c2"]),
        "cdelta": (z, coefficients["cdelta"]),
    }


def unity_ia_bias(
    z: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return a unity IA bias tuple.

    This is useful when the IA amplitudes are already included at the PT power
    spectrum level.

    Args:
        z: Redshift values.

    Returns:
        Tuple ``(z, ones)``.

    Raises:
        ValueError: If any redshift value is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")

    return z, np.ones_like(z, dtype=np.float64)
