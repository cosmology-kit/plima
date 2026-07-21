Nonlinear linear alignment model
================================

Overview
--------

The nonlinear linear alignment (NLA) model is a phenomenological extension of the
linear alignment model.

It retains the assumption that intrinsic galaxy shapes respond linearly to the
large-scale tidal field. However, when constructing the intrinsic alignment
power spectra, the linear matter power spectrum is replaced by a matter power
spectrum that may include nonlinear clustering.

The NLA model therefore introduces nonlinear scale dependence through the
matter power spectrum rather than through a new nonlinear response of galaxy
shapes to the tidal field.

Basic model
-----------

PLIMA defines the NLA response as

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

The matter--intrinsic power spectrum is

.. math::

   P_{\delta I}^{\mathrm{NLA}}(k,z)
   =
   F_{\mathrm{IA}}(z)
   P_{\delta}(k,z),

and the intrinsic--intrinsic power spectrum is

.. math::

   P_{II}^{\mathrm{NLA}}(k,z)
   =
   F_{\mathrm{IA}}^2(z)
   P_{\delta}(k,z).

The supplied :math:`P_{\delta}` is commonly a nonlinear matter power spectrum.
PLIMA deliberately accepts a generic matter power spectrum so that the caller
controls the nonlinear prescription.

Difference between LA and NLA
-----------------------------

The LA and NLA responses have the same form. Their main difference is the
matter power spectrum used in the prediction:

.. math::

   \begin{aligned}
   P_{\delta I}^{\mathrm{LA}}
   &\propto P_{\delta}^{\mathrm{lin}}, \\
   P_{\delta I}^{\mathrm{NLA}}
   &\propto P_{\delta}.
   \end{aligned}

For a nonlinear matter power spectrum,

.. math::

   P_{\delta}
   =
   P_{\delta}^{\mathrm{nonlin}},

the NLA prescription includes nonlinear matter clustering while retaining the
LA tidal response.

.. note::

   NLA is not a complete nonlinear theory of galaxy alignment. It is a useful
   phenomenological prescription obtained by replacing the linear matter power
   spectrum in the LA prediction.

Amplitude prescriptions
-----------------------

PLIMA provides several prescriptions for :math:`A_{\mathrm{IA}}`.

Constant amplitude
~~~~~~~~~~~~~~~~~~

The simplest NLA model assumes

.. math::

   A_{\mathrm{IA}}(z) = A_{\mathrm{IA}}.

This model is registered as ``nla`` and is implemented by
:func:`plima.models.nla.nla_amplitude`.

Redshift-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nla_z`` model uses a linear dependence on scale factor,

.. math::

   A_{\mathrm{IA}}(z)
   =
   A_{\mathrm{IA}}
   +
   b_{\mathrm{IA}}
   \left[
      \frac{a(z)}{a(z_{\mathrm{pivot}})} - 1
   \right],

with

.. math::

   a(z) = \frac{1}{1+z}.

Here :math:`A_{\mathrm{IA}}` is the amplitude at the pivot redshift and
:math:`b_{\mathrm{IA}}` controls the scale-factor dependence.

This model is implemented by
:func:`plima.models.nla.nla_z_amplitude`.

Halo-mass-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nla_m`` model is

.. math::

   A_{\mathrm{IA}}(M)
   =
   A_{\mathrm{IA}}\,
   f_{\mathrm{red}}
   \left(
      \frac{M_{\mathrm{h}}}{M_{\mathrm{pivot}}}
   \right)^{\beta},

where :math:`f_{\mathrm{red}}` is the red-galaxy fraction and :math:`\beta`
controls the halo-mass dependence.

This model is implemented by
:func:`plima.models.nla.nla_mass_amplitude`.

Luminosity-function-dependent amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nla_lf`` model combines luminosity, galaxy type, and redshift evolution,

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

where

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

The factor :math:`\langle w_L(z)\rangle` represents a supplied
luminosity-weighted IA contribution.

The red-galaxy fraction may be supplied directly or calculated from red and
total luminosity functions.

This model is implemented by
:func:`plima.models.nla.lf_nla_amplitude`.

PLIMA implementation
--------------------

The NLA functions are backend independent. Callers supply

- a matter power spectrum,
- a consistently evaluated growth factor,
- :math:`\Omega_{\mathrm{m}}`, and
- an IA amplitude.

The main spectrum functions are

- :func:`plima.models.nla.p_delta_i_nla`, and
- :func:`plima.models.nla.p_ii_nla`.

PLIMA does not decide which nonlinear matter-power prescription should be
used. This choice belongs to the cosmology backend or calling analysis.

References
----------

- Bridle and King (2007), nonlinear extension of the linear alignment
  prescription, arXiv:0705.0166.
- Hirata and Seljak (2004), linear tidal-alignment framework,
  arXiv:astro-ph/0406275.