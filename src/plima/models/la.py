"""Linear alignment intrinsic alignment models.

This module provides small composable functions for linear alignment LA
intrinsic alignment calculations. The amplitude functions define different
redshift, halo mass, or luminosity function dependent IA normalizations. The
spectrum functions then convert those amplitudes into matter intrinsic and
intrinsic intrinsic power spectra using the standard LA response.

The functions are backend independent. They do not require CCL or any other
cosmology package. Callers provide the linear matter power spectrum, growth
factor, and cosmological density parameter explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plima.models.model_registry import register_ia_model
from plima.utils.constants import (
    C1_RHO_CRITICAL,
    DEFAULT_HIGH_Z_PIVOT_REDSHIFT,
    DEFAULT_LOW_Z_PIVOT_REDSHIFT,
    DEFAULT_PIVOT_HALO_MASS,
    DEFAULT_PIVOT_REDSHIFT,
)
from plima.utils.converters import redshift_to_scale_factor
from plima.utils.validators import (
    as_finite_float_array,
    validate_fraction,
    validate_greater_than,
    validate_positive,
)

__all__ = [
    "la_amplitude",
    "la_mass_amplitude",
    "la_z_amplitude",
    "red_fraction_from_luminosity_functions",
    "lf_la_amplitude",
    "la_response",
    "p_delta_i_la",
    "p_ii_la",
]


@register_ia_model(
    "la",
    aliases=(
        "constant_la",
        "la_constant",
        "linear_alignment",
    ),
)
def la_amplitude(
    z: ArrayLike,
    *,
    a_ia: float,
) -> NDArray[np.float64]:
    """Return a constant LA amplitude on a redshift grid.

    Args:
        z: Redshift values where the amplitude should be evaluated.
        a_ia: Constant intrinsic alignment amplitude.

    Returns:
        Array with the same shape as ``z`` containing the constant amplitude.

    Raises:
        ValueError: If any redshift value is not finite.
    """
    z = as_finite_float_array(z, name="z")

    return np.full_like(z, float(a_ia), dtype=np.float64)


@register_ia_model(
    "la_z",
    aliases=(
        "z_la",
        "redshift_la",
        "la_redshift",
        "linear_alignment_redshift",
    ),
)
def la_z_amplitude(
    z: ArrayLike,
    *,
    a_ia: float,
    b_ia: float,
    pivot_redshift: float = DEFAULT_PIVOT_REDSHIFT,
) -> NDArray[np.float64]:
    """Return the KiDS style redshift dependent LA amplitude.

    Args:
        z: Redshift values where the amplitude should be evaluated. Values
            may be negative for future scale factors, but must be greater
            than ``-1``.
        a_ia: Intrinsic alignment amplitude at the pivot redshift.
        b_ia: Linear scale factor dependence of the IA amplitude.
        pivot_redshift: Pivot redshift used to define the redshift dependence.
            This may be negative, but must be greater than ``-1``.

    Returns:
        Redshift dependent LA amplitude evaluated at ``z``.

    Raises:
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If ``pivot_redshift`` is less than or equal to ``-1``.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")
    validate_greater_than(
        pivot_redshift,
        threshold=-1.0,
        name="pivot_redshift",
    )

    scale_factor = redshift_to_scale_factor(z)
    pivot_scale_factor = redshift_to_scale_factor(pivot_redshift)

    return a_ia + b_ia * (scale_factor / pivot_scale_factor - 1.0)


@register_ia_model(
    "la_m",
    aliases=(
        "m_la",
        "mass_la",
        "la_mass",
        "halo_mass_la",
        "linear_alignment_mass",
    ),
)
def la_mass_amplitude(
    *,
    a_ia: float,
    red_fraction: ArrayLike,
    halo_mass: ArrayLike,
    beta: float,
    pivot_halo_mass: float = DEFAULT_PIVOT_HALO_MASS,
) -> NDArray[np.float64]:
    """Return a galaxy type and halo mass dependent LA amplitude.

    Args:
        a_ia: Overall intrinsic alignment amplitude.
        red_fraction: Red galaxy fraction for each sample, bin, or object.
        halo_mass: Halo mass values.
        beta: Power law dependence of the IA amplitude on halo mass.
        pivot_halo_mass: Pivot halo mass used to normalize the mass scaling.

    Returns:
        LA amplitude weighted by red fraction and halo mass.

    Raises:
        ValueError: If ``red_fraction`` is outside the interval from zero to one.
        ValueError: If any halo mass value is not positive.
        ValueError: If ``pivot_halo_mass`` is not positive.
    """
    validate_positive(pivot_halo_mass, name="pivot_halo_mass")

    red_fraction = as_finite_float_array(red_fraction, name="red_fraction")
    halo_mass = as_finite_float_array(halo_mass, name="halo_mass")

    validate_fraction(red_fraction, name="red_fraction")
    validate_positive(halo_mass, name="halo_mass")

    return a_ia * red_fraction * (halo_mass / pivot_halo_mass) ** beta


def la_response(
    growth_factor: ArrayLike,
    omega_m: float,
    *,
    amplitude: ArrayLike | float,
    c1_rho_critical: float = C1_RHO_CRITICAL,
) -> NDArray[np.float64]:
    """Return the LA response multiplying the linear matter power spectrum.

    Args:
        growth_factor: Linear growth factor evaluated at the redshifts of
            interest.
        omega_m: Present day matter density fraction.
        amplitude: LA amplitude. This may be a scalar or an array matching
            the shape of ``growth_factor``.
        c1_rho_critical: Conventional LA normalization ``C1 * rho_crit``.

    Returns:
        LA response factor used in ``P_deltaI = response * P_linear``.

    Raises:
        ValueError: If any growth factor value is not positive.
        ValueError: If ``omega_m`` is not positive.
        ValueError: If ``c1_rho_critical`` is not positive.
    """
    growth_factor = as_finite_float_array(growth_factor, name="growth_factor")
    amplitude = as_finite_float_array(amplitude, name="amplitude")

    validate_positive(growth_factor, name="growth_factor")
    validate_positive(omega_m, name="omega_m")
    validate_positive(c1_rho_critical, name="c1_rho_critical")

    prefactor = c1_rho_critical * omega_m / growth_factor

    return -amplitude * prefactor


def p_delta_i_la(
    linear_matter_power: ArrayLike,
    growth_factor: ArrayLike,
    omega_m: float,
    *,
    amplitude: ArrayLike | float,
    c1_rho_critical: float = C1_RHO_CRITICAL,
) -> NDArray[np.float64]:
    """Return the LA matter intrinsic power spectrum.

    Args:
        linear_matter_power: Linear matter power spectrum values.
        growth_factor: Linear growth factor evaluated consistently with
            ``linear_matter_power``.
        omega_m: Present day matter density fraction.
        amplitude: LA amplitude. This may be a scalar or an array matching
            ``linear_matter_power``.
        c1_rho_critical: Conventional LA normalization ``C1 * rho_crit``.

    Returns:
        Matter intrinsic power spectrum predicted by the LA model.

    Raises:
        ValueError: If any linear matter power value is not finite.
        ValueError: If any growth factor value is not positive.
    """
    linear_matter_power = as_finite_float_array(
        linear_matter_power,
        name="linear_matter_power",
    )

    response = la_response(
        growth_factor,
        omega_m,
        amplitude=amplitude,
        c1_rho_critical=c1_rho_critical,
    )

    return response * linear_matter_power


def p_ii_la(
    linear_matter_power: ArrayLike,
    growth_factor: ArrayLike,
    omega_m: float,
    *,
    amplitude: ArrayLike | float,
    c1_rho_critical: float = C1_RHO_CRITICAL,
) -> NDArray[np.float64]:
    """Return the LA intrinsic intrinsic power spectrum.

    Args:
        linear_matter_power: Linear matter power spectrum values.
        growth_factor: Linear growth factor evaluated consistently with
            ``linear_matter_power``.
        omega_m: Present day matter density fraction.
        amplitude: LA amplitude. This may be a scalar or an array matching
            ``linear_matter_power``.
        c1_rho_critical: Conventional LA normalization ``C1 * rho_crit``.

    Returns:
        Intrinsic intrinsic power spectrum predicted by the LA model.

    Raises:
        ValueError: If any linear matter power value is not finite.
        ValueError: If any growth factor value is not positive.
    """
    linear_matter_power = as_finite_float_array(
        linear_matter_power,
        name="linear_matter_power",
    )

    response = la_response(
        growth_factor,
        omega_m,
        amplitude=amplitude,
        c1_rho_critical=c1_rho_critical,
    )

    return response**2 * linear_matter_power


def red_fraction_from_luminosity_functions(
    z: ArrayLike,
    *,
    red_lf: Any,
    all_lf: Any,
    m_bright: float,
    m_faint: float,
    n_m: int = 512,
) -> NDArray[np.float64]:
    """Return the red galaxy fraction using LFKit luminosity functions.

    Args:
        z: Redshift values where the red fraction should be evaluated.
        red_lf: LFKit luminosity function for the red galaxy population.
        all_lf: LFKit luminosity function or callable for the full galaxy
            population.
        m_bright: Bright absolute magnitude limit.
        m_faint: Faint absolute magnitude limit.
        n_m: Number of magnitude samples used by LFKit for the integral.

    Returns:
        Red galaxy fraction evaluated at each redshift.

    Raises:
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If ``m_faint`` is not greater than ``m_bright``.
        ValueError: If ``n_m`` is less than two.
        ValueError: If LFKit returns values outside the interval from zero to one.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")
    validate_greater_than(m_faint, threshold=m_bright, name="m_faint")
    validate_greater_than(n_m, threshold=1, name="n_m")

    fraction = np.empty_like(z, dtype=np.float64)

    for index, redshift in np.ndenumerate(z):
        fraction[index] = red_lf.fractions.red_fraction(
            float(redshift),
            all_lf,
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=int(n_m),
        )

    validate_fraction(fraction, name="red_fraction")

    return fraction


@register_ia_model(
    "la_lf",
    aliases=(
        "lf_la",
        "luminosity_la",
        "luminosity_function_la",
        "la_luminosity_function",
        "linear_alignment_luminosity_function",
    ),
)
def lf_la_amplitude(
    z: ArrayLike,
    luminosity_weighted_average: ArrayLike,
    *,
    a_ia: float,
    red_fraction: ArrayLike | float = 1.0,
    eta_low_z: float = 0.0,
    eta_high_z: float = 0.0,
    low_z_pivot: float = DEFAULT_LOW_Z_PIVOT_REDSHIFT,
    high_z_pivot: float = DEFAULT_HIGH_Z_PIVOT_REDSHIFT,
) -> NDArray[np.float64]:
    """Return a luminosity function weighted LA amplitude.

    Args:
        z: Redshift values where the amplitude should be evaluated. Values
            may be negative for future scale factors, but must be greater
            than ``-1``.
        luminosity_weighted_average: Luminosity dependent IA weight, usually
            computed from a luminosity function or galaxy sample.
        a_ia: Overall intrinsic alignment amplitude.
        red_fraction: Red galaxy fraction multiplying the IA amplitude.
        eta_low_z: Power law redshift evolution applied at all redshifts.
        eta_high_z: Additional high redshift power law evolution.
        low_z_pivot: Pivot redshift for the low redshift evolution term. This
            may be negative, but must be greater than ``-1``.
        high_z_pivot: Redshift above which the high redshift evolution term is
            applied. This may be negative, but must be greater than ``-1``.

    Returns:
        Redshift, red fraction, and luminosity weighted LA amplitude.

    Raises:
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If either redshift pivot is less than or equal to ``-1``.
        ValueError: If ``red_fraction`` is outside the interval from zero to one.
    """
    z = as_finite_float_array(z, name="z")
    luminosity_weighted_average = as_finite_float_array(
        luminosity_weighted_average,
        name="luminosity_weighted_average",
    )
    red_fraction = as_finite_float_array(red_fraction, name="red_fraction")

    validate_greater_than(z, threshold=-1.0, name="z")
    validate_greater_than(low_z_pivot, threshold=-1.0, name="low_z_pivot")
    validate_greater_than(high_z_pivot, threshold=-1.0, name="high_z_pivot")
    validate_fraction(red_fraction, name="red_fraction")

    redshift_amplitude = a_ia * np.ones_like(z, dtype=np.float64)
    redshift_amplitude *= ((1.0 + z) / (1.0 + low_z_pivot)) ** eta_low_z
    redshift_amplitude *= np.where(
        z > high_z_pivot,
        ((1.0 + z) / (1.0 + high_z_pivot)) ** eta_high_z,
        1.0,
    )

    return redshift_amplitude * red_fraction * luminosity_weighted_average
