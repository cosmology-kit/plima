"""CCL backend helpers for TATT intrinsic alignment power spectra.

This module connects PLIMA TATT amplitude models to CCL PT tracers and Pk2D
objects.

The model normalization lives in ``plima.models.tatt``. This file only handles
CCL specific objects such as ``PTIntrinsicAlignmentTracer``,
``PTMatterTracer``, ``EulerianPTCalculator``, ``WeakLensingTracer``, and
``Pk2D``.

For TATT, IA amplitudes are included in the perturbation theory ``Pk2D``
objects. These spectra should be passed to ``ccl.angular_cl`` through
``p_of_k_a``. The IA only weak lensing tracer should use a unity IA bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
from numpy.typing import ArrayLike

from plima.models.tatt import tatt_pt_biases
from plima.utils.constants import C1_RHO_CRITICAL, DEFAULT_PIVOT_REDSHIFT
from plima.utils.converters import redshift_to_scale_factor
from plima.utils.validators import as_finite_float_array, validate_greater_than

__all__ = [
    "make_ccl_tatt_pt_tracer",
    "make_ccl_matter_pt_tracer",
    "make_ccl_eulerian_pt_calculator",
    "make_ccl_ia_only_wl_tracer",
    "make_ccl_tatt_pk2d",
    "make_ccl_tatt_power_spectra",
]


def make_ccl_tatt_pt_tracer(
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
):
    """Return a CCL PT intrinsic alignment tracer for TATT.

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
        CCL ``PTIntrinsicAlignmentTracer``.
    """
    biases = tatt_pt_biases(
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

    return pt.PTIntrinsicAlignmentTracer(
        c1=biases["c1"],
        c2=biases["c2"],
        cdelta=biases["cdelta"],
    )


def make_ccl_matter_pt_tracer():
    """Return a CCL PT matter tracer."""
    return pt.PTMatterTracer()


def make_ccl_eulerian_pt_calculator(
    *,
    with_number_counts: bool = False,
    with_ia: bool = True,
    log10k_min: float = -4.0,
    log10k_max: float = 2.0,
    nk_per_decade: int = 20,
    **kwargs: Any,
):
    """Return a CCL Eulerian PT calculator.

    Args:
        with_number_counts: Whether to initialize number count PT terms.
        with_ia: Whether to initialize IA PT terms.
        log10k_min: Minimum ``log10(k)`` for PT sampling.
        log10k_max: Maximum ``log10(k)`` for PT sampling.
        nk_per_decade: Number of k samples per decade.
        **kwargs: Extra keyword arguments passed to CCL.

    Returns:
        CCL ``EulerianPTCalculator``.
    """
    return pt.EulerianPTCalculator(
        with_NC=with_number_counts,
        with_IA=with_ia,
        log10k_min=log10k_min,
        log10k_max=log10k_max,
        nk_per_decade=nk_per_decade,
        **kwargs,
    )


def make_ccl_ia_only_wl_tracer(
    cosmo,
    z: ArrayLike,
    nz: ArrayLike,
):
    """Return a CCL weak lensing tracer containing only IA response.

    This tracer is intended for TATT PT spectra. The IA amplitude is set to
    unity because the TATT amplitudes are already included in the ``Pk2D``
    object passed to ``ccl.angular_cl``.

    Args:
        cosmo: CCL cosmology object.
        z: Redshift values for the source distribution.
        nz: Source redshift distribution values.

    Returns:
        CCL ``WeakLensingTracer`` with ``has_shear=False`` and unity IA bias.

    Raises:
        ValueError: If any redshift value is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If ``nz`` does not have the same shape as ``z``.
    """
    z = as_finite_float_array(z, name="z")
    nz = as_finite_float_array(nz, name="nz")

    validate_greater_than(z, threshold=-1.0, name="z")

    if nz.shape != z.shape:
        msg = "nz must have the same shape as z."
        raise ValueError(msg)

    return ccl.WeakLensingTracer(
        cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z, dtype=np.float64)),
        use_A_ia=False,
    )


def make_ccl_tatt_pk2d(
    *,
    pt_calculator,
    ia_tracer,
    matter_tracer=None,
    return_ia_bb: bool = False,
) -> dict[str, Any]:
    """Return CCL Pk2D objects for TATT matter IA and IA IA spectra.

    Args:
        pt_calculator: CCL PT calculator.
        ia_tracer: CCL PT intrinsic alignment tracer.
        matter_tracer: Optional CCL PT matter tracer. If ``None``, one is made.
        return_ia_bb: Whether to also compute the IA B mode auto spectrum.

    Returns:
        Dictionary containing CCL ``Pk2D`` objects.
    """
    if matter_tracer is None:
        matter_tracer = make_ccl_matter_pt_tracer()

    pk_mi = pt_calculator.get_biased_pk2d(
        matter_tracer,
        tracer2=ia_tracer,
    )
    pk_ii = pt_calculator.get_biased_pk2d(
        ia_tracer,
        tracer2=ia_tracer,
    )

    spectra = {
        "matter_intrinsic": pk_mi,
        "intrinsic_intrinsic": pk_ii,
    }

    if return_ia_bb:
        spectra["intrinsic_intrinsic_bb"] = pt_calculator.get_biased_pk2d(
            ia_tracer,
            tracer2=ia_tracer,
            return_ia_bb=True,
        )

    return spectra


def make_ccl_tatt_power_spectra(
    cosmo,
    z: ArrayLike,
    *,
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
    pt_calculator=None,
    matter_tracer=None,
    return_ia_bb: bool = False,
    calculator_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return CCL TATT PT tracers and Pk2D spectra.

    Args:
        cosmo: CCL cosmology object.
        z: Redshift values used for IA bias sampling.
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
        pt_calculator: Optional existing CCL PT calculator.
        matter_tracer: Optional existing CCL matter PT tracer.
        return_ia_bb: Whether to also compute the IA B mode auto spectrum.
        calculator_kwargs: Keyword arguments used if a PT calculator is made.

    Returns:
        Dictionary containing the IA tracer, matter tracer, PT calculator, and
        TATT ``Pk2D`` spectra.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")

    scale_factor = redshift_to_scale_factor(z)
    growth_factor = ccl.growth_factor(cosmo, scale_factor)
    omega_m = cosmo["Omega_m"]

    ia_tracer = make_ccl_tatt_pt_tracer(
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

    if matter_tracer is None:
        matter_tracer = make_ccl_matter_pt_tracer()

    if pt_calculator is None:
        calculator_kwargs = dict(calculator_kwargs or {})
        pt_calculator = make_ccl_eulerian_pt_calculator(**calculator_kwargs)

    pt_calculator.update_ingredients(cosmo)

    spectra = make_ccl_tatt_pk2d(
        pt_calculator=pt_calculator,
        ia_tracer=ia_tracer,
        matter_tracer=matter_tracer,
        return_ia_bb=return_ia_bb,
    )

    return {
        "ia_tracer": ia_tracer,
        "matter_tracer": matter_tracer,
        "pt_calculator": pt_calculator,
        "spectra": spectra,
    }
