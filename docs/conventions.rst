Conventions
===========

PLIMA follows a small set of conventions that are used consistently across all
intrinsic-alignment models. This page summarizes those conventions and
highlights where they differ from alternative conventions found in the
literature.

Cosmological quantities
-----------------------

Unless stated otherwise,

- :math:`z` denotes redshift,
- :math:`a=(1+z)^{-1}` is the scale factor,
- :math:`D(z)` is the linear growth factor normalized by the cosmology
  backend,
- :math:`\Omega_{\mathrm{m}}` is the present-day matter density parameter, and
- :math:`C_1\rho_{\mathrm{crit}}` is the conventional intrinsic-alignment
  normalization constant.

PLIMA does not compute these quantities internally. They are supplied by the
chosen cosmology backend.

Amplitude conventions
---------------------

PLIMA separates **amplitude models** from **power-spectrum calculations**.

Amplitude models describe how intrinsic-alignment parameters evolve with
redshift, halo mass, luminosity, or galaxy population. The resulting amplitudes
are then passed to a backend that constructs the corresponding IA power
spectra.

This separation allows the same astrophysical prescription to be used with
multiple cosmology backends.

Redshift evolution
------------------

Whenever a model includes redshift evolution using a pivot redshift, PLIMA uses

.. math::

   A(z)
   =
   A_{\rm pivot}
   \left(
      \frac{1+z}{1+z_{\rm pivot}}
   \right)^\eta.

The pivot redshift is configurable for every model.

Scale-factor evolution
----------------------

Some observational analyses instead parameterize the amplitude using the scale
factor.

PLIMA follows the KiDS convention,

.. math::

   A(z)
   =
   A_{\rm pivot}
   +
   b
   \left(
      \frac{a(z)}
           {a(z_{\rm pivot})}
      -1
   \right),

where

.. math::

   a(z)=\frac{1}{1+z}.

This parameterization is implemented by the ``*_z`` LA and NLA models.

Halo-mass scaling
-----------------

Mass-dependent models use

.. math::

   A(M)
   =
   A_{\rm IA}
   f_{\rm red}
   \left(
      \frac{M_{\rm halo}}
           {M_{\rm pivot}}
   \right)^\beta,

where

- :math:`f_{\rm red}` is the red-galaxy fraction,
- :math:`M_{\rm halo}` is the halo mass,
- :math:`M_{\rm pivot}` is the pivot halo mass, and
- :math:`\beta` controls the mass scaling.

Luminosity-function models
--------------------------

The luminosity-function models multiply the intrinsic-alignment amplitude by

- the red-galaxy fraction,
- a luminosity-weighted average supplied by LFKit, and
- optional low- and high-redshift evolution factors.

PLIMA intentionally leaves the luminosity-function calculation itself to
LFKit and only consumes its outputs.

Growth-factor normalization
---------------------------

The LA and NLA responses use

.. math::

   F_{\rm IA}(z)
   =
   -
   A(z)
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}
   }{
      D(z)
   }.

Consequently,

.. math::

   P_{\delta I}
   =
   F_{\rm IA}P,

and

.. math::

   P_{II}
   =
   F_{\rm IA}^2P,

where :math:`P` is either the linear matter power spectrum (LA) or the supplied
matter power spectrum (NLA).

Sign convention
---------------

PLIMA explicitly includes the conventional minus sign in the IA response,

.. math::

   F_{\rm IA}
   =
   -
   A(z)
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}
   }{
      D(z)
   }.

As a result,

- positive :math:`A_{\rm IA}` produces negative
  :math:`P_{\delta I}`, and
- :math:`P_{II}` remains positive because it depends on
  :math:`F_{\rm IA}^2`.

This convention matches the one adopted internally throughout PLIMA.

Power spectra
-------------

PLIMA distinguishes between the linear and nonlinear alignment models only
through the matter power spectrum entering the calculation.

For the LA model,

.. math::

   P_{\delta I}^{\rm LA}
   \propto
   P_{\rm lin},

where :math:`P_{\rm lin}` is the linear matter power spectrum.

For the NLA model,

.. math::

   P_{\delta I}^{\rm NLA}
   \propto
   P_{\rm matter},

where :math:`P_{\rm matter}` is the matter power spectrum supplied by the
backend, typically including nonlinear clustering.

TATT coefficients
-----------------

The TATT model introduces three perturbative coefficients,

.. math::

   c_1,\qquad
   c_2,\qquad
   c_\delta.

PLIMA computes

.. math::

   c_1
   =
   -
   A_1
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}
   }{
      D
   },

.. math::

   c_\delta
   =
   -
   A_{1\delta}
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}
   }{
      D
   },

and by default

.. math::

   c_2
   =
   5
   A_2
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}
   }{
      D^2
   }.

Alternative :math:`c_2` convention
----------------------------------

Some analyses instead define

.. math::

   c_2
   =
   5
   A_2
   \frac{
      C_1\rho_{\rm crit}\Omega_{\rm m}^2
   }{
      \Omega_{\rm m,fid}D^2
   }.

PLIMA supports both conventions through the
``use_omega_m_squared_for_c2`` option.

Backend independence
--------------------

PLIMA is designed to remain independent of any particular cosmology library.

The package prepares intrinsic-alignment amplitudes and parameters but expects
the backend to provide quantities such as

- matter power spectra,
- growth factors,
- distances,
- transfer functions, and
- projection into observable statistics.

Current support is provided through the CCL backend, while the astrophysical
parameterizations remain backend independent.

Units
-----

Unless explicitly stated otherwise,

- redshift is dimensionless,
- amplitudes are dimensionless,
- growth factors are dimensionless,
- halo masses should use the convention expected by the chosen amplitude
  model, and
- power-spectrum units are inherited from the cosmology backend.

Throughout PLIMA, arrays are evaluated on the supplied redshift grid and are
returned as NumPy ``float64`` arrays.
