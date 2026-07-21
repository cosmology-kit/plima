CCL backend: TATT
=================

This page demonstrates how PLIMA tidal-alignment and tidal-torquing (TATT)
models can be used with the perturbation-theory tools provided by the Core
Cosmology Library (CCL).

PLIMA supplies the redshift-dependent TATT amplitudes and their normalized
perturbation-theory coefficients. CCL then constructs the three-dimensional
matter--intrinsic and intrinsic--intrinsic power spectra and projects them into
angular power spectra.

This example requires CCL's perturbation-theory dependencies, including
``fast-pt``.

TATT amplitudes
---------------

The TATT model contains three intrinsic-alignment amplitudes,

.. math::

   A_1(z), \qquad
   A_2(z), \qquad
   A_{1\delta}(z),

where

- :math:`A_1` controls linear tidal alignment,
- :math:`A_2` controls quadratic tidal torquing, and
- :math:`A_{1\delta}` controls source-density weighting.

PLIMA defines their redshift evolution as

.. math::

   A_X(z)
   =
   A_X(z_{\mathrm{pivot}})
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_X}.

.. plot::
   :include-source:
   :caption: Redshift evolution of the PLIMA TATT amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.tatt import tatt_amplitudes


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   z = np.linspace(0.0, 3.0, 400)

   amplitudes = tatt_amplitudes(
       z,
       a1=1.0,
       a2=0.5,
       a1delta=0.25,
       eta1=-0.5,
       eta2=1.0,
       eta1delta=0.5,
       z_pivot=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       z,
       amplitudes["a1"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$A_1(z)$",
   )
   ax.plot(
       z,
       amplitudes["a2"],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$A_2(z)$",
   )
   ax.plot(
       z,
       amplitudes["a1delta"],
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$A_{1\delta}(z)$",
   )

   ax.axhline(
       0.0,
       color="0.6",
       linewidth=1.0,
   )
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("TATT amplitude")
   ax.set_title("TATT amplitude evolution")
   ax.legend(frameon=False)

   fig.tight_layout()

Normalized TATT coefficients
----------------------------

CCL perturbation-theory tracers use the normalized coefficients

.. math::

   c_1(z), \qquad
   c_2(z), \qquad
   c_{\delta}(z).

PLIMA converts the phenomenological amplitudes into these coefficients using
the CCL growth factor and matter density.

.. plot::
   :include-source:
   :caption: TATT coefficients normalized using a CCL cosmology.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.tatt import tatt_normalized_coefficients


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   cosmology = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.67,
       n_s=0.96,
       sigma8=0.81,
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   z = np.linspace(0.0, 3.0, 400)
   scale_factor = 1.0 / (1.0 + z)

   growth_factor = ccl.growth_factor(
       cosmology,
       scale_factor,
   )

   coefficients = tatt_normalized_coefficients(
       z,
       growth_factor=growth_factor,
       omega_m=cosmology["Omega_m"],
       a1=1.0,
       a2=0.5,
       a1delta=0.25,
       eta1=-0.5,
       eta2=1.0,
       eta1delta=0.5,
       z_pivot=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       z,
       coefficients["c1"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$c_1(z)$",
   )
   ax.plot(
       z,
       coefficients["c2"],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$c_2(z)$",
   )
   ax.plot(
       z,
       coefficients["cdelta"],
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$c_{\delta}(z)$",
   )

   ax.axhline(
       0.0,
       color="0.6",
       linewidth=1.0,
   )
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Normalized coefficient")
   ax.set_title("CCL-normalized TATT coefficients")
   ax.legend(frameon=False)

   fig.tight_layout()

Three-dimensional TATT power spectra
------------------------------------

PLIMA can package the normalized coefficients as ``(z, coefficient)`` tuples
for a CCL :class:`pyccl.nl_pt.PTIntrinsicAlignmentTracer`.

The CCL Eulerian perturbation-theory calculator then constructs

.. math::

   P_{\delta I}(k,z),

the matter--intrinsic power spectrum, and

.. math::

   P_{II}^{EE}(k,z),
   \qquad
   P_{II}^{BB}(k,z),

the intrinsic--intrinsic E- and B-mode power spectra.

.. plot::
   :include-source:
   :caption: Three-dimensional TATT intrinsic-alignment power spectra at zero redshift.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl
   import pyccl.nl_pt as pt

   from plima.models.tatt import tatt_pt_biases


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   cosmology = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.67,
       n_s=0.96,
       sigma8=0.81,
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   z = np.linspace(0.0, 3.0, 256)
   scale_factor = 1.0 / (1.0 + z)

   growth_factor = ccl.growth_factor(
       cosmology,
       scale_factor,
   )

   biases = tatt_pt_biases(
       z,
       growth_factor=growth_factor,
       omega_m=cosmology["Omega_m"],
       a1=1.0,
       a2=0.5,
       a1delta=0.25,
       eta1=-0.5,
       eta2=1.0,
       eta1delta=0.5,
       z_pivot=0.62,
   )

   ia_pt_tracer = pt.PTIntrinsicAlignmentTracer(
       c1=biases["c1"],
       c2=biases["c2"],
       cdelta=biases["cdelta"],
   )

   matter_pt_tracer = pt.PTMatterTracer()

   pt_calculator = pt.EulerianPTCalculator(
       with_IA=True,
       log10k_min=-4,
       log10k_max=2,
       nk_per_decade=20,
   )

   pt_calculator.update_ingredients(
       cosmology,
   )

   pk_delta_i = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=matter_pt_tracer,
   )

   pk_ii_ee = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=ia_pt_tracer,
   )

   pk_ii_bb = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=ia_pt_tracer,
       return_ia_bb=True,
   )

   k = np.geomspace(1.0e-3, 10.0, 300)
   scale_factor_plot = 1.0

   power_delta_i = pk_delta_i(
       k,
       scale_factor_plot,
       cosmology,
   )

   power_ii_ee = pk_ii_ee(
       k,
       scale_factor_plot,
       cosmology,
   )

   power_ii_bb = pk_ii_bb(
       k,
       scale_factor_plot,
       cosmology,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       k,
       np.abs(power_delta_i),
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$|P_{\delta I}|$",
   )
   ax.plot(
       k,
       np.abs(power_ii_ee),
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$|P_{II}^{EE}|$",
   )
   ax.plot(
       k,
       np.abs(power_ii_bb),
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$|P_{II}^{BB}|$",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Wavenumber $k\,[\mathrm{Mpc}^{-1}]$")
   ax.set_ylabel(r"$|P(k)|\,[\mathrm{Mpc}^{3}]$")
   ax.set_title("TATT power spectra at $z=0$")
   ax.legend(frameon=False)

   fig.tight_layout()

Absolute values are displayed because the matter--intrinsic spectrum can be
negative under the adopted IA sign convention.

TATT angular power spectra
--------------------------

The perturbation-theory power spectra can be passed directly to
:func:`pyccl.angular_cl`.

A standard CCL weak-lensing tracer is used for gravitational shear. An
IA-only weak-lensing tracer is constructed with

- ``has_shear=False`` to remove gravitational shear,
- a unity IA bias because the TATT coefficients are already included in the
  perturbation-theory power spectra, and
- ``use_A_ia=False`` to disable CCL's standard LA/NLA normalization.

This produces the angular spectra

.. math::

   C_{\ell}^{GI},

.. math::

   C_{\ell}^{II,EE},

and

.. math::

   C_{\ell}^{II,BB}.

.. plot::
   :include-source:
   :caption: TATT matter--intrinsic and intrinsic--intrinsic angular power spectra.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl
   import pyccl.nl_pt as pt

   from plima.models.tatt import tatt_pt_biases
   from plima.models.tatt import unity_ia_bias


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   cosmology = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.67,
       n_s=0.96,
       sigma8=0.81,
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   z_bias = np.linspace(0.0, 3.0, 256)
   scale_factor = 1.0 / (1.0 + z_bias)

   growth_factor = ccl.growth_factor(
       cosmology,
       scale_factor,
   )

   biases = tatt_pt_biases(
       z_bias,
       growth_factor=growth_factor,
       omega_m=cosmology["Omega_m"],
       a1=1.0,
       a2=0.5,
       a1delta=0.25,
       eta1=-0.5,
       eta2=1.0,
       eta1delta=0.5,
       z_pivot=0.62,
   )

   ia_pt_tracer = pt.PTIntrinsicAlignmentTracer(
       c1=biases["c1"],
       c2=biases["c2"],
       cdelta=biases["cdelta"],
   )

   matter_pt_tracer = pt.PTMatterTracer()

   pt_calculator = pt.EulerianPTCalculator(
       with_IA=True,
       log10k_min=-4,
       log10k_max=2,
       nk_per_decade=20,
   )

   pt_calculator.update_ingredients(
       cosmology,
   )

   pk_delta_i = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=matter_pt_tracer,
   )

   pk_ii_ee = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=ia_pt_tracer,
   )

   pk_ii_bb = pt_calculator.get_biased_pk2d(
       ia_pt_tracer,
       tracer2=ia_pt_tracer,
       return_ia_bb=True,
   )

   z_source = np.linspace(0.01, 3.0, 512)

   nz = z_source**2 * np.exp(
       -(z_source / 0.7) ** 1.5
   )
   nz /= np.trapezoid(
       nz,
       z_source,
   )

   shear_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z_source, nz),
   )

   ia_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z_source, nz),
       has_shear=False,
       ia_bias=unity_ia_bias(z_source),
       use_A_ia=False,
   )

   ell = np.unique(
       np.geomspace(10, 2000, 120).astype(int)
   ).astype(float)

   cell_gi = ccl.angular_cl(
       cosmology,
       shear_tracer,
       ia_tracer,
       ell,
       p_of_k_a=pk_delta_i,
   )

   cell_ii_ee = ccl.angular_cl(
       cosmology,
       ia_tracer,
       ia_tracer,
       ell,
       p_of_k_a=pk_ii_ee,
   )

   cell_ii_bb = ccl.angular_cl(
       cosmology,
       ia_tracer,
       ia_tracer,
       ell,
       p_of_k_a=pk_ii_bb,
   )

   prefactor = ell * (ell + 1.0) / (
       2.0 * np.pi
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       ell,
       prefactor * np.abs(cell_gi),
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$|GI|$",
   )
   ax.plot(
       ell,
       prefactor * np.abs(cell_ii_ee),
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$|II|$ E-mode",
   )
   ax.plot(
       ell,
       prefactor * np.abs(cell_ii_bb),
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$|II|$ B-mode",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Angular multipole $\ell$")
   ax.set_ylabel(
       r"$\ell(\ell+1)|C_\ell|/(2\pi)$"
   )
   ax.set_title("TATT angular power spectra")
   ax.legend(frameon=False)

   fig.tight_layout()

The figure shows absolute values so that the signed :math:`GI` contribution
can be displayed on the same logarithmic axes as the positive
intrinsic--intrinsic spectra.

The B-mode spectrum arises from the nonlinear tidal-torquing contributions.
It vanishes in the pure linear-alignment limit obtained by setting

.. math::

   A_2 = A_{1\delta} = 0.