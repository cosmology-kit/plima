"""Numerical constants used by PLIMA."""

from __future__ import annotations

C1 = 5.0e-14
"""Historical linear-alignment normalization in h^-2 M_sun^-1 Mpc^3."""

RHO_CRITICAL_H2 = 2.775e11
"""Critical density today in h^2 M_sun Mpc^-3."""

C1_RHO_CRITICAL = 0.0134
"""Standard NLA intrinsic-alignment normalization convention."""

DEFAULT_PIVOT_REDSHIFT = 0.62
"""Default pivot redshift from Joachimi et al. 2011, arXiv:1008.3491"""

DEFAULT_LOW_Z_PIVOT_REDSHIFT = 0.3
"""Default low z pivot redshift from Sarcevic et al. 2024, arXiv:2406.03352 and Krause et al. 2016 arXiv:1506.08730"""

DEFAULT_HIGH_Z_PIVOT_REDSHIFT = 0.75
"""Default high z pivot redshift from Sarcevic et al. 2024, arXiv:2406.03352 and Krause et al. 2016 arXiv:1506.08730"""

DEFAULT_PIVOT_HALO_MASS = 10**13.5
"""Default pivot halo mass from Wright et al. 2025 arXiv:2503.19441 and Fortuna et al. 2025 arXiv:2409.15416"""
