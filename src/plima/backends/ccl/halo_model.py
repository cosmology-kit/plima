"""CCL backend helpers for halo model intrinsic alignment power spectra."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import ArrayLike

from plima.utils.constants import C1_RHO_CRITICAL
from plima.utils.types import FloatArray
from plima.utils.validators import (
    as_finite_float_array,
    validate_finite,
    validate_positive,
)

__all__ = [
    "halo_model_nla_prefactor",
    "make_ccl_halo_model_ia_power_spectra",
    "make_ccl_halo_model_windowed_power_spectra",
    "make_ccl_halo_model_power_spectra",
]


def halo_model_nla_prefactor(
    cosmo: Any,
    a: ArrayLike,
    *,
    a_ia: float = 1.0,
    c1_rho_critical: float = C1_RHO_CRITICAL,
) -> FloatArray:
    """Return the NLA intrinsic alignment prefactor.

    Args:
        cosmo: CCL cosmology object.
        a: Scale factor values.
        a_ia: Intrinsic alignment amplitude.
        c1_rho_critical: IA normalization constant.

    Returns:
        NLA prefactor evaluated at each scale factor.

    Raises:
        ValueError: If any numerical input is not finite.
        ValueError: If any scale factor value is not positive.
        ValueError: If scale factor values are not one dimensional.
    """
    a = as_finite_float_array(a, name="a")

    if a.ndim != 1:
        msg = "a must be one dimensional."
        raise ValueError(msg)

    validate_positive(a, name="a")
    validate_finite(a_ia, name="a_ia")
    validate_finite(c1_rho_critical, name="c1_rho_critical")

    return np.asarray(
        a_ia * c1_rho_critical * cosmo["Omega_m"] / cosmo.growth_factor(a),
        dtype=np.float64,
    )


def _pk2d_from_array(
    *,
    k: FloatArray,
    a: FloatArray,
    pk: FloatArray,
) -> Any:
    """Build a CCL ``Pk2D`` object from an array."""
    pk = np.asarray(pk, dtype=np.float64)

    expected_shape = (a.size, k.size)
    if pk.shape != expected_shape:
        msg = f"pk must have shape {expected_shape}, got {pk.shape}."
        raise ValueError(msg)

    if not np.all(np.isfinite(pk)):
        msg = "pk must contain only finite values."
        raise ValueError(msg)

    return ccl.pk2d.Pk2D(
        a_arr=a,
        lk_arr=np.log(k),
        pk_arr=pk,
        is_logp=False,
    )


def _power_grid(
    values: ArrayLike,
    *,
    k: FloatArray,
    a: FloatArray,
    name: str,
) -> FloatArray:
    """Return a power spectrum grid with shape ``(len(a), len(k))``."""
    grid = np.asarray(values, dtype=np.float64)

    expected_shape = (a.size, k.size)
    transposed_shape = (k.size, a.size)

    if grid.shape == expected_shape:
        output = grid
    elif grid.shape == transposed_shape:
        output = grid.T
    else:
        msg = f"{name} must have shape {expected_shape}, got {grid.shape}."
        raise ValueError(msg)

    if not np.all(np.isfinite(output)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)

    return output


def _evaluate_pk2d_grid(
    pk2d: Any,
    *,
    k: FloatArray,
    a: FloatArray,
    name: str,
) -> FloatArray:
    """Evaluate a ``Pk2D`` object on a grid."""
    return _power_grid(pk2d(k, a), k=k, a=a, name=name)


def make_ccl_halo_model_ia_power_spectra(
    cosmo: Any,
    *,
    k: ArrayLike,
    a: ArrayLike,
    a_ia: float = 1.0,
    a1h: float = 0.001,
    b: float = -2.0,
    mass_def: str = "200m",
    use_linear_2h: bool = True,
) -> dict[str, Any]:
    """Return CCL halo model IA power spectrum components.

    Args:
        cosmo: CCL cosmology object.
        k: Wavenumber array in CCL units.
        a: Scale factor array.
        a_ia: Large scale intrinsic alignment amplitude.
        a1h: Satellite one halo alignment amplitude.
        b: Satellite shear radial power law slope.
        mass_def: Halo mass definition passed to CCL.
        use_linear_2h: If True, use linear matter power for central two halo
            terms.

    Returns:
        Dictionary containing CCL ``Pk2D`` component spectra.

    Raises:
        ValueError: If any numerical input is not finite.
        ValueError: If any wavenumber or scale factor value is not positive.
    """
    k = as_finite_float_array(k, name="k")
    a = as_finite_float_array(a, name="a")

    if k.ndim != 1:
        msg = "k must be one dimensional."
        raise ValueError(msg)

    if a.ndim != 1:
        msg = "a must be one dimensional."
        raise ValueError(msg)

    validate_positive(k, name="k")
    validate_positive(a, name="a")
    validate_finite(a_ia, name="a_ia")
    validate_finite(a1h, name="a1h")
    validate_finite(b, name="b")

    concentration = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    mass_function = ccl.halos.MassFuncTinker10(mass_def=mass_def)
    halo_bias = ccl.halos.HaloBiasTinker10(mass_def=mass_def)

    hmc = ccl.halos.HMCalculator(
        mass_function=mass_function,
        halo_bias=halo_bias,
        mass_def=mass_def,
    )

    satellite_shear = ccl.halos.SatelliteShearHOD(
        concentration=concentration,
        mass_def=mass_def,
        a1h=a1h,
        b=b,
    )

    matter_profile = ccl.halos.HaloProfileNFW(
        mass_def=mass_def,
        concentration=concentration,
        truncated=True,
        fourier_analytic=True,
    )

    c_ia = halo_model_nla_prefactor(cosmo, a, a_ia=a_ia)

    if use_linear_2h:
        matter_power = cosmo.linear_matter_power(k, a)
    else:
        matter_power = cosmo.nonlin_matter_power(k, a)

    matter_power = _power_grid(
        matter_power,
        k=k,
        a=a,
        name="matter_power",
    )

    c_pk_lin = _pk2d_from_array(
        k=k,
        a=a,
        pk=c_ia.reshape(-1, 1) * matter_power,
    )

    ii_1h_ss = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        satellite_shear,
        get_2h=False,
        a_arr=a,
        lk_arr=np.log(k),
    )

    ii_2h_ss = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        satellite_shear,
        get_1h=False,
        a_arr=a,
        lk_arr=np.log(k),
    )

    bias_gamma = ccl.halos.halomod_bias_1pt(
        cosmo,
        hmc,
        k,
        a,
        satellite_shear,
    )
    bias_gamma = _power_grid(
        bias_gamma,
        k=k,
        a=a,
        name="bias_gamma",
    )

    pk_b_gamma = _pk2d_from_array(
        k=k,
        a=a,
        pk=-bias_gamma,
    )

    ii_2h_cs = c_pk_lin * pk_b_gamma

    ii_2h_cc = _pk2d_from_array(
        k=k,
        a=a,
        pk=c_ia.reshape(-1, 1) ** 2 * matter_power,
    )

    gi_1h_s = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        matter_profile,
        prof2=satellite_shear,
        get_2h=False,
        a_arr=a,
        lk_arr=np.log(k),
    )

    gi_2h_s = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        matter_profile,
        prof2=satellite_shear,
        get_1h=False,
        a_arr=a,
        lk_arr=np.log(k),
    )

    gi_2h_c = -1.0 * c_pk_lin

    return {
        "ii_1h_ss": ii_1h_ss,
        "ii_2h_ss": ii_2h_ss,
        "ii_2h_cs": ii_2h_cs,
        "ii_2h_cc": ii_2h_cc,
        "ii_total": ii_1h_ss + ii_2h_ss + ii_2h_cs + ii_2h_cc,
        "gi_1h_s": gi_1h_s,
        "gi_2h_s": gi_2h_s,
        "gi_2h_c": gi_2h_c,
        "gi_total": gi_1h_s + gi_2h_s + gi_2h_c,
    }


def make_ccl_halo_model_windowed_power_spectra(
    cosmo: Any,
    *,
    k: ArrayLike,
    a: ArrayLike,
    ia_power_spectra: dict[str, Any],
    a_ia: float = 1.0,
    k_1h: float | None = None,
    k_2h: float | None = None,
) -> dict[str, Any]:
    """Return windowed CCL halo model IA spectra.

    Args:
        cosmo: CCL cosmology object.
        k: Wavenumber array in CCL units.
        a: Scale factor array.
        ia_power_spectra: Output from
            ``make_ccl_halo_model_ia_power_spectra``.
        a_ia: Large scale intrinsic alignment amplitude.
        k_1h: One halo transition scale. If None, use ``4 * cosmo["h"]``.
        k_2h: Two halo damping scale. If None, use ``6 * cosmo["h"]``.

    Returns:
        Dictionary containing combined spectra and components.

    Raises:
        KeyError: If required spectra are missing.
        ValueError: If any numerical input is not finite.
        ValueError: If any wavenumber, scale factor, or transition scale is not
            positive.
    """
    k = as_finite_float_array(k, name="k")
    a = as_finite_float_array(a, name="a")

    if k.ndim != 1:
        msg = "k must be one dimensional."
        raise ValueError(msg)

    if a.ndim != 1:
        msg = "a must be one dimensional."
        raise ValueError(msg)

    validate_positive(k, name="k")
    validate_positive(a, name="a")
    validate_finite(a_ia, name="a_ia")

    if k_1h is None:
        k_1h = 4.0 * cosmo["h"]
    if k_2h is None:
        k_2h = 6.0 * cosmo["h"]

    validate_positive(k_1h, name="k_1h")
    validate_positive(k_2h, name="k_2h")

    c_ia = halo_model_nla_prefactor(cosmo, a, a_ia=a_ia)
    matter_power = _power_grid(
        cosmo.nonlin_matter_power(k, a),
        k=k,
        a=a,
        name="matter_power",
    )

    two_halo_window = np.exp(-((k / k_2h) ** 2)).reshape(1, -1)
    one_halo_window = (1.0 - np.exp(-((k / k_1h) ** 2))).reshape(1, -1)

    ii_nla = _pk2d_from_array(
        k=k,
        a=a,
        pk=c_ia.reshape(-1, 1) ** 2 * matter_power * two_halo_window,
    )

    gi_nla = _pk2d_from_array(
        k=k,
        a=a,
        pk=-c_ia.reshape(-1, 1) * matter_power * two_halo_window,
    )

    ii_1h_values = _evaluate_pk2d_grid(
        ia_power_spectra["ii_1h_ss"],
        k=k,
        a=a,
        name="ii_1h_ss",
    )
    gi_1h_values = _evaluate_pk2d_grid(
        ia_power_spectra["gi_1h_s"],
        k=k,
        a=a,
        name="gi_1h_s",
    )

    ii_1h = _pk2d_from_array(
        k=k,
        a=a,
        pk=ii_1h_values * one_halo_window,
    )

    gi_1h = _pk2d_from_array(
        k=k,
        a=a,
        pk=gi_1h_values * one_halo_window,
    )

    return {
        "ii": ii_nla + ii_1h,
        "gi": gi_nla + gi_1h,
        "ii_nla": ii_nla,
        "gi_nla": gi_nla,
        "ii_1h": ii_1h,
        "gi_1h": gi_1h,
    }


def make_ccl_halo_model_power_spectra(
    cosmo: Any,
    *,
    k: ArrayLike,
    a: ArrayLike,
    a_ia: float = 1.0,
    a1h: float = 0.001,
    b: float = -2.0,
    k_1h: float | None = None,
    k_2h: float | None = None,
    mass_def: str = "200m",
    use_linear_2h: bool = True,
) -> dict[str, Any]:
    """Return CCL halo model IA components and windowed spectra.

    Args:
        cosmo: CCL cosmology object.
        k: Wavenumber array in CCL units.
        a: Scale factor array.
        a_ia: Large scale intrinsic alignment amplitude.
        a1h: Satellite one halo alignment amplitude.
        b: Satellite shear radial power law slope.
        k_1h: One halo transition scale.
        k_2h: Two halo damping scale.
        mass_def: Halo mass definition passed to CCL.
        use_linear_2h: If True, use linear matter power for central two halo
            terms.

    Returns:
        Dictionary containing raw components and windowed spectra.
    """
    components = make_ccl_halo_model_ia_power_spectra(
        cosmo,
        k=k,
        a=a,
        a_ia=a_ia,
        a1h=a1h,
        b=b,
        mass_def=mass_def,
        use_linear_2h=use_linear_2h,
    )

    spectra = make_ccl_halo_model_windowed_power_spectra(
        cosmo,
        k=k,
        a=a,
        ia_power_spectra=components,
        a_ia=a_ia,
        k_1h=k_1h,
        k_2h=k_2h,
    )

    return {
        "components": components,
        "spectra": spectra,
    }
