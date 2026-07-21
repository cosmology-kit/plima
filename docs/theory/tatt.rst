Tidal alignment and tidal torquing model
========================================

Overview
--------

The tidal alignment and tidal torquing (TATT) model extends the linear
alignment picture by including additional responses of galaxy shapes to the
large-scale tidal and density fields.

At leading order, the intrinsic shape responds linearly to the tidal field.
TATT adds a quadratic tidal response and a contribution produced by weighting
the tidal-alignment field by the local source density.

Schematically, the intrinsic shape tensor may be expanded as

.. math::

   \gamma_{ij}^{I}
   =
   c_1 s_{ij}
   +
   c_2
   \left(
      s_{ik}s_{kj}
      -
      \frac{1}{3}\delta_{ij}s_{kl}s_{kl}
   \right)
   +
   c_{\delta}\,\delta s_{ij}
   + \cdots,

where

- :math:`s_{ij}` is the traceless tidal field,
- :math:`\delta` is the matter overdensity,
- :math:`c_1` is the linear tidal-alignment coefficient,
- :math:`c_2` is the quadratic tidal-torquing coefficient, and
- :math:`c_{\delta}` is the source-density-weighting coefficient.

The resulting matter--intrinsic and intrinsic--intrinsic spectra contain
several perturbative contributions and are not generally proportional to a
single matter power spectrum.

TATT amplitudes
---------------

PLIMA first defines dimensionless amplitudes at a pivot redshift. Their
redshift evolution is

.. math::

   A_1(z)
   =
   a_1
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_1},

.. math::

   A_2(z)
   =
   a_2
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_2},

and

.. math::

   A_{1\delta}(z)
   =
   a_{1\delta}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{1\delta}}.

The amplitudes describe three different physical contributions:

``a1``
   The linear tidal-alignment amplitude at the pivot redshift.

``a2``
   The quadratic tidal-torquing amplitude at the pivot redshift.

``a1delta``
   The source-density-weighting amplitude at the pivot redshift.

The corresponding ``eta`` parameters control their redshift evolution.

These amplitudes are returned by
:func:`plima.models.tatt.tatt_amplitudes`.

Perturbative coefficients
-------------------------

The dimensionless amplitudes are converted into the coefficients required by a
perturbative IA tracer.

Linear tidal-alignment coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLIMA defines

.. math::

   c_1(z)
   =
   -A_1(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}
        {D(z)}.

This term is the TATT equivalent of the usual LA or NLA response.

Source-density-weighting coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The density-weighting coefficient is

.. math::

   c_{\delta}(z)
   =
   -A_{1\delta}(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}
        {D(z)}.

This term describes the coupling between the linear tidal-alignment field and
the local source density.

Quadratic tidal-torquing coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, PLIMA uses

.. math::

   c_2(z)
   =
   5 A_2(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}
        {D^2(z)}.

PLIMA also supports the alternative convention

.. math::

   c_2(z)
   =
   5 A_2(z)
   \frac{
      C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}^2
   }{
      \Omega_{\mathrm{m,fid}}D^2(z)
   }.

The alternative form is selected with
``use_omega_m_squared_for_c2=True``.

Because both conventions appear in the literature and in analysis pipelines,
the convention used for :math:`c_2` should always be recorded.

Relation to simpler models
--------------------------

The linear tidal-alignment part is recovered by setting

.. math::

   a_2 = 0,
   \qquad
   a_{1\delta} = 0.

In this limit only :math:`c_1` remains. The resulting response has the same
large-scale structure as the LA model, although the full power-spectrum
calculation is performed through the perturbation-theory backend.

A nonzero :math:`a_2` introduces the quadratic tidal-torquing contribution,
while a nonzero :math:`a_{1\delta}` introduces source-density weighting.

PLIMA implementation
--------------------

The TATT module prepares amplitudes and normalized perturbative coefficients.
It does not calculate the complete TATT power spectra itself.

The primary helpers are

- :func:`plima.models.tatt.tatt_amplitudes`,
- :func:`plima.models.tatt.tatt_normalized_coefficients`, and
- :func:`plima.models.tatt.tatt_pt_biases`.

The PT bias helper returns

.. code-block:: text

   {
       "c1": (z, c1),
       "c2": (z, c2),
       "cdelta": (z, cdelta),
   }

These values may be passed to a perturbation-theory IA tracer such as CCL's
``PTIntrinsicAlignmentTracer``.

Unity tracer bias
-----------------

In some pipelines, the IA amplitudes are already included directly in the
three-dimensional power spectra. In that case, multiplying by the amplitude
again at the tracer level would double count the IA response.

The helper :func:`plima.models.tatt.unity_ia_bias` returns

.. math::

   b_{\mathrm{IA}}(z) = 1,

on the supplied redshift grid. This is useful when the downstream tracer needs
an IA-bias input but the physical amplitude is already contained in the
power-spectrum object.

References
----------

- Blazek et al. (2019), *Beyond linear galaxy alignments*,
  arXiv:1708.09247.
- CCL ``PTIntrinsicAlignmentTracer`` and ``translate_IA_norm``
  documentation.