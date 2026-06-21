"""Unit tests for ``plima.backends.ccl.tatt``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.backends.ccl import tatt as ccl_tatt


class FakePTIntrinsicAlignmentTracer:
    """Fake CCL PT intrinsic alignment tracer."""

    def __init__(self, *, c1, c2, cdelta) -> None:
        """Stores PT intrinsic alignment bias inputs."""
        self.c1 = c1
        self.c2 = c2
        self.cdelta = cdelta


class FakePTMatterTracer:
    """Fake CCL PT matter tracer."""


class FakeEulerianPTCalculator:
    """Fake CCL Eulerian PT calculator."""

    def __init__(self, **kwargs) -> None:
        """Stores calculator keyword arguments."""
        self.kwargs = kwargs
        self.updated_cosmo = None
        self.calls = []

    def update_ingredients(self, cosmo) -> None:
        """Records the cosmology used to update PT ingredients."""
        self.updated_cosmo = cosmo

    def get_biased_pk2d(self, tracer1, *, tracer2, return_ia_bb=False):
        """Returns fake Pk2D labels and records the requested tracer pair."""
        call = {
            "tracer1": tracer1,
            "tracer2": tracer2,
            "return_ia_bb": return_ia_bb,
        }
        self.calls.append(call)

        if return_ia_bb:
            return "pk_bb"

        if isinstance(tracer1, FakePTMatterTracer):
            return "pk_mi"

        return "pk_ii"


class FakeWeakLensingTracer:
    """Fake CCL weak lensing tracer."""

    def __init__(
        self,
        cosmo,
        *,
        dndz,
        has_shear,
        ia_bias,
        use_A_ia,
    ) -> None:
        """Stores weak lensing tracer inputs."""
        self.cosmo = cosmo
        self.dndz = dndz
        self.has_shear = has_shear
        self.ia_bias = ia_bias
        self.use_A_ia = use_A_ia


def test_make_ccl_tatt_pt_tracer_builds_ia_tracer(monkeypatch) -> None:
    """Tests that the TATT helper builds a CCL PT IA tracer."""
    monkeypatch.setattr(
        ccl_tatt.pt,
        "PTIntrinsicAlignmentTracer",
        FakePTIntrinsicAlignmentTracer,
    )

    z = np.array([0.1, 0.5, 1.0])
    growth_factor = np.array([0.95, 0.75, 0.55])

    tracer = ccl_tatt.make_ccl_tatt_pt_tracer(
        z,
        growth_factor=growth_factor,
        omega_m=0.3,
        a1=1.2,
        a2=0.4,
        a1delta=0.7,
        eta1=0.1,
        eta2=0.2,
        eta1delta=0.3,
        z_pivot=0.5,
    )

    assert isinstance(tracer, FakePTIntrinsicAlignmentTracer)

    z_c1, c1 = tracer.c1
    z_c2, c2 = tracer.c2
    z_cdelta, cdelta = tracer.cdelta

    np.testing.assert_allclose(z_c1, z)
    np.testing.assert_allclose(z_c2, z)
    np.testing.assert_allclose(z_cdelta, z)

    assert c1.shape == z.shape
    assert c2.shape == z.shape
    assert cdelta.shape == z.shape


def test_make_ccl_matter_pt_tracer(monkeypatch) -> None:
    """Tests that the matter PT helper builds a CCL PT matter tracer."""
    monkeypatch.setattr(ccl_tatt.pt, "PTMatterTracer", FakePTMatterTracer)

    tracer = ccl_tatt.make_ccl_matter_pt_tracer()

    assert isinstance(tracer, FakePTMatterTracer)


def test_make_ccl_eulerian_pt_calculator(monkeypatch) -> None:
    """Tests that the Eulerian PT helper forwards calculator options."""
    monkeypatch.setattr(
        ccl_tatt.pt,
        "EulerianPTCalculator",
        FakeEulerianPTCalculator,
    )

    calculator = ccl_tatt.make_ccl_eulerian_pt_calculator(
        with_number_counts=True,
        with_ia=False,
        log10k_min=-3.0,
        log10k_max=1.0,
        nk_per_decade=10,
        extra_option="value",
    )

    assert isinstance(calculator, FakeEulerianPTCalculator)
    assert calculator.kwargs == {
        "with_NC": True,
        "with_IA": False,
        "log10k_min": -3.0,
        "log10k_max": 1.0,
        "nk_per_decade": 10,
        "extra_option": "value",
    }


def test_make_ccl_ia_only_wl_tracer_uses_unity_ia_bias(monkeypatch) -> None:
    """Tests that the IA only weak lensing tracer uses unity IA bias."""
    monkeypatch.setattr(
        ccl_tatt.ccl,
        "WeakLensingTracer",
        FakeWeakLensingTracer,
    )

    cosmo = {"Omega_m": 0.3}
    z = np.array([0.1, 0.5, 1.0])
    nz = np.array([0.2, 0.5, 0.3])

    tracer = ccl_tatt.make_ccl_ia_only_wl_tracer(cosmo, z, nz)

    assert isinstance(tracer, FakeWeakLensingTracer)
    assert tracer.cosmo is cosmo
    assert tracer.has_shear is False
    assert tracer.use_A_ia is False

    dndz_z, dndz_nz = tracer.dndz
    ia_z, ia_bias = tracer.ia_bias

    np.testing.assert_allclose(dndz_z, z)
    np.testing.assert_allclose(dndz_nz, nz)
    np.testing.assert_allclose(ia_z, z)
    np.testing.assert_allclose(ia_bias, np.ones_like(z))


def test_make_ccl_ia_only_wl_tracer_rejects_mismatched_shapes(
    monkeypatch,
) -> None:
    """Tests that the IA only weak lensing tracer rejects shape mismatch."""
    monkeypatch.setattr(
        ccl_tatt.ccl,
        "WeakLensingTracer",
        FakeWeakLensingTracer,
    )

    cosmo = {"Omega_m": 0.3}
    z = np.array([0.1, 0.5, 1.0])
    nz = np.array([0.2, 0.5])

    with pytest.raises(ValueError, match="nz must have the same shape as z"):
        ccl_tatt.make_ccl_ia_only_wl_tracer(cosmo, z, nz)


def test_make_ccl_tatt_pk2d_returns_expected_spectra(monkeypatch) -> None:
    """Tests that TATT Pk2D construction returns all requested spectra."""
    monkeypatch.setattr(ccl_tatt.pt, "PTMatterTracer", FakePTMatterTracer)

    calculator = FakeEulerianPTCalculator()
    matter_tracer = FakePTMatterTracer()
    ia_tracer = FakePTIntrinsicAlignmentTracer(
        c1=(np.array([0.1]), np.array([1.0])),
        c2=(np.array([0.1]), np.array([2.0])),
        cdelta=(np.array([0.1]), np.array([3.0])),
    )

    spectra = ccl_tatt.make_ccl_tatt_pk2d(
        pt_calculator=calculator,
        ia_tracer=ia_tracer,
        matter_tracer=matter_tracer,
        return_ia_bb=True,
    )

    assert spectra == {
        "matter_intrinsic": "pk_mi",
        "intrinsic_intrinsic": "pk_ii",
        "intrinsic_intrinsic_bb": "pk_bb",
    }

    assert len(calculator.calls) == 3
    assert calculator.calls[0]["tracer1"] is matter_tracer
    assert calculator.calls[0]["tracer2"] is ia_tracer
    assert calculator.calls[0]["return_ia_bb"] is False

    assert calculator.calls[1]["tracer1"] is ia_tracer
    assert calculator.calls[1]["tracer2"] is ia_tracer
    assert calculator.calls[1]["return_ia_bb"] is False

    assert calculator.calls[2]["tracer1"] is ia_tracer
    assert calculator.calls[2]["tracer2"] is ia_tracer
    assert calculator.calls[2]["return_ia_bb"] is True


def test_make_ccl_tatt_pk2d_makes_default_matter_tracer(monkeypatch) -> None:
    """Tests that TATT Pk2D construction creates a default matter tracer."""
    monkeypatch.setattr(ccl_tatt.pt, "PTMatterTracer", FakePTMatterTracer)

    calculator = FakeEulerianPTCalculator()
    ia_tracer = FakePTIntrinsicAlignmentTracer(
        c1=(np.array([0.1]), np.array([1.0])),
        c2=(np.array([0.1]), np.array([2.0])),
        cdelta=(np.array([0.1]), np.array([3.0])),
    )

    spectra = ccl_tatt.make_ccl_tatt_pk2d(
        pt_calculator=calculator,
        ia_tracer=ia_tracer,
    )

    assert spectra == {
        "matter_intrinsic": "pk_mi",
        "intrinsic_intrinsic": "pk_ii",
    }
    assert isinstance(calculator.calls[0]["tracer1"], FakePTMatterTracer)


def test_make_ccl_tatt_power_spectra_builds_everything(monkeypatch) -> None:
    """Tests that the high level TATT helper builds tracers and spectra."""
    monkeypatch.setattr(ccl_tatt.pt, "PTMatterTracer", FakePTMatterTracer)
    monkeypatch.setattr(
        ccl_tatt.pt,
        "PTIntrinsicAlignmentTracer",
        FakePTIntrinsicAlignmentTracer,
    )
    monkeypatch.setattr(
        ccl_tatt.pt,
        "EulerianPTCalculator",
        FakeEulerianPTCalculator,
    )

    def fake_growth_factor(cosmo, scale_factor):
        """Returns fake growth values and checks scale factor conversion."""
        assert cosmo == {"Omega_m": 0.3}
        np.testing.assert_allclose(scale_factor, np.array([1.0, 2.0 / 3.0]))
        return np.array([1.0, 0.8])

    monkeypatch.setattr(ccl_tatt.ccl, "growth_factor", fake_growth_factor)

    cosmo = {"Omega_m": 0.3}
    result = ccl_tatt.make_ccl_tatt_power_spectra(
        cosmo,
        [0.0, 0.5],
        a1=1.0,
        a2=0.5,
        a1delta=0.2,
        calculator_kwargs={"log10k_min": -3.0},
    )

    assert isinstance(result["ia_tracer"], FakePTIntrinsicAlignmentTracer)
    assert isinstance(result["matter_tracer"], FakePTMatterTracer)
    assert isinstance(result["pt_calculator"], FakeEulerianPTCalculator)
    assert result["pt_calculator"].updated_cosmo is cosmo

    assert result["spectra"] == {
        "matter_intrinsic": "pk_mi",
        "intrinsic_intrinsic": "pk_ii",
    }

    assert result["pt_calculator"].kwargs["log10k_min"] == -3.0


def test_make_ccl_tatt_power_spectra_rejects_invalid_redshift() -> None:
    """Tests that the high level TATT helper rejects invalid redshifts."""
    cosmo = {"Omega_m": 0.3}

    with pytest.raises(ValueError, match="z"):
        ccl_tatt.make_ccl_tatt_power_spectra(cosmo, [-1.0, 0.5])


def test_make_ccl_tatt_pt_tracer_matches_tatt_amplitudes(monkeypatch) -> None:
    """Tests that the TATT PT tracer stores the expected amplitude arrays."""
    monkeypatch.setattr(
        ccl_tatt.pt,
        "PTIntrinsicAlignmentTracer",
        FakePTIntrinsicAlignmentTracer,
    )

    z = np.array([0.0, 0.5, 1.0])
    growth_factor = np.array([1.0, 0.8, 0.6])

    tracer = ccl_tatt.make_ccl_tatt_pt_tracer(
        z,
        growth_factor=growth_factor,
        omega_m=0.3,
        a1=1.0,
        a2=2.0,
        a1delta=3.0,
        eta1=0.0,
        eta2=0.0,
        eta1delta=0.0,
        z_pivot=0.5,
    )

    _, c1 = tracer.c1
    _, c2 = tracer.c2
    _, cdelta = tracer.cdelta

    assert c1.shape == z.shape
    assert c2.shape == z.shape
    assert cdelta.shape == z.shape
    assert np.all(np.isfinite(c1))
    assert np.all(np.isfinite(c2))
    assert np.all(np.isfinite(cdelta))


def test_make_ccl_tatt_pt_tracer_reduces_to_linear_alignment_limit(
    monkeypatch,
) -> None:
    """Tests that TATT reduces to linear alignment when nonlinear terms vanish."""
    monkeypatch.setattr(
        ccl_tatt.pt,
        "PTIntrinsicAlignmentTracer",
        FakePTIntrinsicAlignmentTracer,
    )

    z = np.array([0.0, 0.5, 1.0])
    growth_factor = np.array([1.0, 0.8, 0.6])

    tracer = ccl_tatt.make_ccl_tatt_pt_tracer(
        z,
        growth_factor=growth_factor,
        omega_m=0.3,
        a1=1.0,
        a2=0.0,
        a1delta=0.0,
        eta1=0.0,
        eta2=0.0,
        eta1delta=0.0,
        z_pivot=0.5,
    )

    z_c1, c1 = tracer.c1
    z_c2, c2 = tracer.c2
    z_cdelta, cdelta = tracer.cdelta

    np.testing.assert_allclose(z_c1, z)
    np.testing.assert_allclose(z_c2, z)
    np.testing.assert_allclose(z_cdelta, z)

    assert np.all(np.isfinite(c1))
    assert np.any(c1 != 0.0)
    np.testing.assert_allclose(c2, np.zeros_like(z))
    np.testing.assert_allclose(cdelta, np.zeros_like(z))
