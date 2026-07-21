Linear alignment model
======================

Overview
--------

The linear alignment (LA) model describes intrinsic galaxy shapes as a linear
response to the large-scale tidal field.

In this picture, the tidal gravitational field present during galaxy formation
produces a preferred orientation for galaxy shapes. Because the same matter
field also produces gravitational lensing, intrinsic galaxy shapes can become
correlated both with each other and with the lensing shear.

The LA model is primarily a large-scale model. It assumes that the intrinsic
shape field responds linearly to the tidal field and uses the linear matter
power spectrum when constructing intrinsic alignment power spectra.

Basic model
-----------

PLIMA writes the redshift-dependent intrinsic alignment response as

.. math::

   F_{\mathrm{IA}}(z)
   =
   -A_{\mathrm{IA}}(z)
   \frac{C_1 \rho_{\mathrm{crit}} \Omega_{\mathrm{m}}}
        {D(z)},

where

- :math:`A_{\mathrm{IA}}(z)` is the intrinsic alignment amplitude,
- :math:`D(z)` is the supplied linear growth factor,
- :math:`\Omega_{\mathrm{m}}` is the present-day matter density fraction, and
- :math:`C_1\rho_{\mathrm{crit}}` is the conventional IA normalization.

The matter--intrinsic and intrinsic--intrinsic power spectra are then

.. math::

   P_{\delta I}^{\mathrm{LA}}(k,z)
   =
   F_{\mathrm{IA}}(z)
   P_{\delta}^{\mathrm{lin}}(k,z),

and

.. math::

   P_{II}^{\mathrm{LA}}(k,z)
   =
   F_{\mathrm{IA}}^2(z)
   P_{\delta}^{\mathrm{lin}}(k,z).

Here :math:`P_{\delta}^{\mathrm{lin}}` is the linear matter power spectrum.

.. note::

   With the convention used by PLIMA, a positive
   :math:`A_{\mathrm{IA}}` produces a negative
   :math:`P_{\delta I}` because the minus sign is included explicitly in
   :math:`F_{\mathrm{IA}}`.

Amplitude prescriptions
-----------------------

PLIMA provides several prescriptions for :math:`A_{\mathrm{IA}}`.

Constant amplitude
~~~~~~~~~~~~~~~~~~

The simplest model assumes a constant amplitude,

.. math::

   A_{\mathrm{IA}}(z) = A_{\mathrm{IA}}.

This model is registered as ``la`` and is implemented by
:func:`plima.models.la.la_amplitude`.

Redshift-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``la_z`` model uses a linear dependence on scale factor,

.. math::

   A_{\mathrm{IA}}(z)
   =
   A_{\mathrm{IA}}
   +
   b_{\mathrm{IA}}
   \left[
      \frac{a(z)}{a(z_{\mathrm{pivot}})} - 1
   \right],

where

.. math::

   a(z) = \frac{1}{1+z}.

The parameter :math:`A_{\mathrm{IA}}` is the amplitude at the pivot redshift,
while :math:`b_{\mathrm{IA}}` controls its scale-factor dependence.

This model is implemented by
:func:`plima.models.la.la_z_amplitude`.

Halo-mass-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``la_m`` model allows the alignment amplitude to depend on galaxy type and
halo mass,

.. math::

   A_{\mathrm{IA}}(M)
   =
   A_{\mathrm{IA}}\,
   f_{\mathrm{red}}
   \left(
      \frac{M_{\mathrm{h}}}{M_{\mathrm{pivot}}}
   \right)^{\beta},

where

- :math:`f_{\mathrm{red}}` is the red-galaxy fraction,
- :math:`M_{\mathrm{h}}` is the halo mass,
- :math:`M_{\mathrm{pivot}}` is the pivot halo mass, and
- :math:`\beta` controls the halo-mass scaling.

This model is implemented by
:func:`plima.models.la.la_mass_amplitude`.

Luminosity-function-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``la_lf`` model combines redshift evolution, the red-galaxy fraction, and
a luminosity-dependent weight,

.. math::

   A_{\mathrm{IA}}^{\mathrm{LF}}(z)
   =
   A_{\mathrm{IA}}\,
   f_{\mathrm{red}}(z)\,
   \langle w_L(z) \rangle
   \left(
      \frac{1+z}{1+z_{\mathrm{low}}}
   \right)^{\eta_{\mathrm{low}}}
   H_{\mathrm{high}}(z),

where :math:`\langle w_L(z)\rangle` is the supplied luminosity-weighted
average. The high-redshift factor is

.. math::

   H_{\mathrm{high}}(z)
   =
   \begin{cases}
   \left(
      \dfrac{1+z}{1+z_{\mathrm{high}}}
   \right)^{\eta_{\mathrm{high}}},
   & z > z_{\mathrm{high}}, \\[8pt]
   1,
   & z \leq z_{\mathrm{high}}.
   \end{cases}

The red-galaxy fraction may be supplied directly or evaluated from red and
total luminosity functions.

This model is implemented by
:func:`plima.models.la.lf_la_amplitude`.

PLIMA implementation
--------------------

The LA implementation is backend independent. PLIMA does not calculate the
matter power spectrum or growth factor internally. Instead, callers provide

- the linear matter power spectrum,
- the growth factor,
- :math:`\Omega_{\mathrm{m}}`, and
- the desired amplitude prescription.

The main spectrum helpers are

- :func:`plima.models.la.p_delta_i_la`, and
- :func:`plima.models.la.p_ii_la`.

These functions calculate three-dimensional IA power spectra. Projection into
angular power spectra or correlation functions is handled by a cosmology
backend such as CCL.

References
----------

- Catelan, Kamionkowski, and Blandford (2001), linear tidal-alignment model.
- Hirata and Seljak (2004), intrinsic alignment--lensing interference,
  arXiv:astro-ph/0406275.