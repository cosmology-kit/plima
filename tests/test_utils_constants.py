"""Unit tests for ``plima.utils.constants``."""

from __future__ import annotations

import numpy as np

from plima.utils.constants import (
    C1,
    C1_RHO_CRITICAL,
    DEFAULT_HIGH_Z_PIVOT_REDSHIFT,
    DEFAULT_LOW_Z_PIVOT_REDSHIFT,
    DEFAULT_PIVOT_HALO_MASS,
    DEFAULT_PIVOT_REDSHIFT,
    RHO_CRITICAL_H2,
)


def test_constants_values() -> None:
    """Tests that numerical constants keep their expected values."""
    assert C1 == 5.0e-14
    assert RHO_CRITICAL_H2 == 2.775e11
    assert C1_RHO_CRITICAL == 0.0134
    assert DEFAULT_PIVOT_REDSHIFT == 0.62
    assert DEFAULT_LOW_Z_PIVOT_REDSHIFT == 0.3
    assert DEFAULT_HIGH_Z_PIVOT_REDSHIFT == 0.75
    assert DEFAULT_PIVOT_HALO_MASS == 10**13.5


def test_c1_rho_critical_convention() -> None:
    """Tests that the NLA normalization is a convention."""
    product = C1 * RHO_CRITICAL_H2

    assert np.isclose(product, 0.013875)
    assert C1_RHO_CRITICAL == 0.0134
    assert not np.isclose(product, C1_RHO_CRITICAL, rtol=1.0e-4)
