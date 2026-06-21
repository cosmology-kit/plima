"""Unit tests for ``plima.backends.ccl.la``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.backends.ccl.la import make_ccl_la_ia_bias


def test_make_ccl_la_ia_bias_uses_default_amplitude() -> None:
    """Tests that LA IA bias defaults to the requested amplitude."""
    z = np.array([0.0, 0.5, 1.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, a_ia=2.5)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(ia_bias, np.full_like(z, 2.5))
    assert z_out.dtype == np.float64
    assert ia_bias.dtype == np.float64


def test_make_ccl_la_ia_bias_uses_precomputed_amplitude() -> None:
    """Tests that LA IA bias uses a precomputed amplitude when supplied."""
    z = np.array([0.0, 0.5, 1.0])
    amplitude = np.array([1.0, 2.0, 3.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(ia_bias, amplitude)


def test_make_ccl_la_ia_bias_accepts_scalar_amplitude_for_scalar_z() -> None:
    """Tests that LA IA bias accepts scalar amplitude for scalar redshift."""
    z_out, ia_bias = make_ccl_la_ia_bias(0.5, amplitude=2.0)

    np.testing.assert_allclose(z_out, np.array([0.5]))
    np.testing.assert_allclose(ia_bias, np.array([2.0]))
    assert z_out.shape == (1,)
    assert ia_bias.shape == (1,)


def test_make_ccl_la_ia_bias_accepts_scalar_amplitude_for_array_z() -> None:
    """Tests that LA IA bias broadcasts scalar amplitude over redshift arrays."""
    z = np.array([0.0, 0.5, 1.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=2.0)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(ia_bias, np.array([2.0, 2.0, 2.0]))


def test_make_ccl_la_ia_bias_rejects_amplitude_shape_mismatch() -> None:
    """Tests that LA IA bias rejects amplitudes with a mismatched shape."""
    z = np.array([0.0, 0.5, 1.0])
    amplitude = np.array([1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="amplitude must be scalar or broadcastable to the same shape as z.",
    ):
        make_ccl_la_ia_bias(z, amplitude=amplitude)


@pytest.mark.parametrize(
    "z",
    [
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, -np.inf]),
    ],
)
def test_make_ccl_la_ia_bias_rejects_nonfinite_redshift(z: np.ndarray) -> None:
    """Tests that LA IA bias rejects nonfinite redshift values."""
    with pytest.raises(ValueError, match="z must contain only finite values."):
        make_ccl_la_ia_bias(z)


@pytest.mark.parametrize(
    "amplitude",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_make_ccl_la_ia_bias_rejects_nonfinite_amplitude(
    amplitude: np.ndarray,
) -> None:
    """Tests that LA IA bias rejects nonfinite amplitude values."""
    z = np.array([0.0, 0.5])

    with pytest.raises(ValueError, match="amplitude must contain only finite values."):
        make_ccl_la_ia_bias(z, amplitude=amplitude)


@pytest.mark.parametrize(
    "z",
    [
        np.array([-1.0]),
        np.array([-1.1]),
        np.array([0.0, -1.0]),
    ],
)
def test_make_ccl_la_ia_bias_rejects_invalid_redshift_domain(z: np.ndarray) -> None:
    """Tests that LA IA bias rejects redshift values less than or equal to minus one."""
    with pytest.raises(
        ValueError,
        match=r"z must contain only values greater than -1\.0\.",
    ):
        make_ccl_la_ia_bias(z)


def test_make_ccl_la_ia_bias_preserves_redshift_and_amplitude_order() -> None:
    """Tests that LA IA bias preserves the input redshift and amplitude order."""
    z = np.array([1.0, 0.0, 0.5])
    amplitude = np.array([3.0, 1.0, 2.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(ia_bias, amplitude)


def test_make_ccl_la_ia_bias_returns_copies_as_float_arrays() -> None:
    """Tests that LA IA bias returns float arrays independent of integer inputs."""
    z = np.array([1, 0])
    amplitude = np.array([2, 1])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    np.testing.assert_allclose(z_out, np.array([1.0, 0.0]))
    np.testing.assert_allclose(ia_bias, np.array([2.0, 1.0]))
    assert z_out.dtype == np.float64
    assert ia_bias.dtype == np.float64


def test_make_ccl_la_ia_bias_accepts_list_redshift_and_amplitude() -> None:
    """Tests that LA IA bias accepts list inputs."""
    z = [0.0, 0.5, 1.0]
    amplitude = [1.0, 2.0, 3.0]

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    np.testing.assert_allclose(z_out, np.array(z))
    np.testing.assert_allclose(ia_bias, np.array(amplitude))
    assert z_out.dtype == np.float64
    assert ia_bias.dtype == np.float64


def test_make_ccl_la_ia_bias_returns_two_arrays() -> None:
    """Tests that LA IA bias returns the CCL tuple structure."""
    result = make_ccl_la_ia_bias([0.0, 0.5], amplitude=[1.0, 2.0])

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)


def test_make_ccl_la_ia_bias_returns_arrays_with_matching_shapes() -> None:
    """Tests that LA IA bias returns arrays with matching shapes."""
    z = np.array([0.0, 0.5, 1.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=2.0)

    assert z_out.shape == ia_bias.shape
    assert z_out.shape == z.shape


def test_make_ccl_la_ia_bias_returns_independent_arrays() -> None:
    """Tests that LA IA bias returns arrays independent of input arrays."""
    z = np.array([0.0, 0.5, 1.0])
    amplitude = np.array([1.0, 2.0, 3.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    z[0] = 99.0
    amplitude[0] = 99.0

    np.testing.assert_allclose(z_out, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(ia_bias, np.array([1.0, 2.0, 3.0]))


def test_make_ccl_la_ia_bias_rejects_empty_redshift() -> None:
    """Tests that LA IA bias rejects empty redshift arrays."""
    with pytest.raises(ValueError, match="z must contain at least one value."):
        make_ccl_la_ia_bias(np.array([]))


def test_make_ccl_la_ia_bias_rejects_empty_amplitude_for_nonempty_redshift() -> None:
    """Tests that LA IA bias rejects empty amplitudes for nonempty redshift arrays."""
    z = np.array([0.0, 0.5])

    with pytest.raises(
        ValueError,
        match="amplitude must be scalar or broadcastable to the same shape as z.",
    ):
        make_ccl_la_ia_bias(z, amplitude=np.array([]))


def test_make_ccl_la_ia_bias_accepts_length_one_amplitude_array() -> None:
    """Tests that LA IA bias accepts length one amplitude arrays."""
    z = np.array([0.0, 0.5, 1.0])
    amplitude = np.array([2.0])

    z_out, ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    np.testing.assert_allclose(z_out, z)
    np.testing.assert_allclose(ia_bias, np.full_like(z, 2.0))
