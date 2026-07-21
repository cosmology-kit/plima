CCL backend: LA
===============

This page demonstrates how the PLIMA linear-alignment amplitude prescriptions
can be used with the Core Cosmology Library (CCL).

PLIMA supplies the intrinsic-alignment amplitude, while CCL supplies the
cosmology, linear matter power spectrum, weak-lensing kernels, and angular
projection.

Shared setup
------------

All examples use the same cosmology, source redshift distribution, and
multipole range.

Constant LA model
-----------------

The constant LA model uses

.. math::

   A_{\mathrm{IA}}(z) = A_{\mathrm{IA}}.

The PLIMA amplitude is passed directly to the CCL ``ia_bias`` argument. The
linear CCL matter power spectrum is supplied explicitly to ``angular_cl`` so
that the calculation follows the LA prescription.

.. plot::
   :include-source:
   :caption: Linear-alignment angular spectra for a constant IA amplitude.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.la import la_amplitude


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"
   PLIMA_GREEN = "#5EB23B"

   cosmology = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.67,
       n_s=0.96,
       sigma8=0.81,
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   z = np.linspace(0.01, 3.0, 400)

   nz = z**2 * np.exp(-(z / 0.7) ** 1.5)
   nz /= np.trapezoid(nz, z)

   amplitude = la_amplitude(
       z,
       a_ia=1.0,
   )

   shear_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
   )

   ia_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       has_shear=False,
       ia_bias=(z, amplitude),
   )

   observed_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       ia_bias=(z, amplitude),
   )

   ell = np.unique(
       np.geomspace(10, 3000, 120).astype(int)
   )

   linear_power = cosmology.get_linear_power()

   cell_gg = ccl.angular_cl(
       cosmology,
       shear_tracer,
       shear_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_gi_ig = 2.0 * ccl.angular_cl(
       cosmology,
       shear_tracer,
       ia_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_ii = ccl.angular_cl(
       cosmology,
       ia_tracer,
       ia_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_total = ccl.angular_cl(
       cosmology,
       observed_tracer,
       observed_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   prefactor = ell * (ell + 1.0) / (2.0 * np.pi)

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       ell,
       prefactor * cell_gg,
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$GG$",
   )
   ax.plot(
       ell,
       -prefactor * cell_gi_ig,
       color=PLIMA_RED,
       linewidth=2.5,
       linestyle="--",
       label=r"$-(GI+IG)$",
   )
   ax.plot(
       ell,
       prefactor * cell_ii,
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$II$",
   )
   ax.plot(
       ell,
       prefactor * cell_total,
       color=PLIMA_GREEN,
       linewidth=3.0,
       label=r"$GG+GI+IG+II$",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Angular multipole $\ell$")
   ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
   ax.set_title("Constant LA angular spectra")
   ax.legend(frameon=False)

   fig.tight_layout()

The negative cross contribution is plotted as :math:`-(GI+IG)` so that it can
be displayed on logarithmic axes.

Redshift-dependent LA model
---------------------------

The ``la_z`` prescription introduces a linear dependence on scale factor.

.. plot::
   :include-source:
   :caption: Effect of the redshift-dependent LA prescription on the observed spectrum.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.la import la_amplitude, la_z_amplitude


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

   z = np.linspace(0.01, 3.0, 400)

   nz = z**2 * np.exp(-(z / 0.7) ** 1.5)
   nz /= np.trapezoid(nz, z)

   amplitude_constant = la_amplitude(
       z,
       a_ia=1.0,
   )

   amplitude_redshift = la_z_amplitude(
       z,
       a_ia=1.0,
       b_ia=-1.5,
       pivot_redshift=0.62,
   )

   shear_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
   )

   constant_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       ia_bias=(z, amplitude_constant),
   )

   redshift_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       ia_bias=(z, amplitude_redshift),
   )

   ell = np.unique(
       np.geomspace(10, 3000, 120).astype(int)
   )

   linear_power = cosmology.get_linear_power()

   cell_gg = ccl.angular_cl(
       cosmology,
       shear_tracer,
       shear_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_constant = ccl.angular_cl(
       cosmology,
       constant_tracer,
       constant_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_redshift = ccl.angular_cl(
       cosmology,
       redshift_tracer,
       redshift_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   prefactor = ell * (ell + 1.0) / (2.0 * np.pi)

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       ell,
       prefactor * cell_gg,
       color=PLIMA_BLUE,
       linewidth=2.2,
       linestyle=":",
       label="No IA",
   )
   ax.plot(
       ell,
       prefactor * cell_constant,
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label="Constant LA",
   )
   ax.plot(
       ell,
       prefactor * cell_redshift,
       color=PLIMA_RED,
       linewidth=2.8,
       label="Redshift-dependent LA",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Angular multipole $\ell$")
   ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
   ax.set_title("Redshift-dependent LA model")
   ax.legend(frameon=False)

   fig.tight_layout()

Halo-mass-dependent LA model
----------------------------

The ``la_m`` prescription weights the IA amplitude by red-galaxy fraction and
halo mass. In this example, smooth effective halo-mass and red-fraction curves
are assigned to the source sample.

.. plot::
   :include-source:
   :caption: LA spectrum with a halo-mass-dependent amplitude.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.la import la_mass_amplitude


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

   z = np.linspace(0.01, 3.0, 400)

   nz = z**2 * np.exp(-(z / 0.7) ** 1.5)
   nz /= np.trapezoid(nz, z)

   red_fraction = 0.8 * np.exp(-0.25 * z)
   halo_mass = 1.0e13 * (1.0 + z) ** 0.8

   amplitude = la_mass_amplitude(
       a_ia=1.0,
       red_fraction=red_fraction,
       halo_mass=halo_mass,
       beta=0.5,
       pivot_halo_mass=1.0e13,
   )

   shear_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
   )

   mass_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       ia_bias=(z, amplitude),
   )

   ell = np.unique(
       np.geomspace(10, 3000, 120).astype(int)
   )

   linear_power = cosmology.get_linear_power()

   cell_gg = ccl.angular_cl(
       cosmology,
       shear_tracer,
       shear_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_mass = ccl.angular_cl(
       cosmology,
       mass_tracer,
       mass_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   prefactor = ell * (ell + 1.0) / (2.0 * np.pi)

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       ell,
       prefactor * cell_gg,
       color=PLIMA_BLUE,
       linewidth=2.5,
       linestyle="--",
       label="No IA",
   )
   ax.plot(
       ell,
       prefactor * cell_mass,
       color=PLIMA_RED,
       linewidth=2.8,
       label="Mass-dependent LA",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Angular multipole $\ell$")
   ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
   ax.set_title("Halo-mass-dependent LA model")
   ax.legend(frameon=False)

   fig.tight_layout()

Luminosity-function-dependent LA model
--------------------------------------

The ``la_lf`` prescription combines a red-galaxy fraction, a
luminosity-weighted contribution, and redshift evolution.

.. plot::
   :include-source:
   :caption: LA spectrum with a luminosity-function-dependent amplitude.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.la import lf_la_amplitude


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   cosmology = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.67,
       n_s=0.96,
       sigma8=0.81,
       transfer_function="eisenstein_hu",
       matter_power_spectrum="halofit",
   )

   z = np.linspace(0.01, 3.0, 400)

   nz = z**2 * np.exp(-(z / 0.7) ** 1.5)
   nz /= np.trapezoid(nz, z)

   red_fraction = 0.8 * np.exp(-0.25 * z)
   luminosity_weighted_average = 0.8 + 0.45 * z

   amplitude = lf_la_amplitude(
       z,
       luminosity_weighted_average,
       a_ia=1.0,
       red_fraction=red_fraction,
       eta_low_z=0.5,
       eta_high_z=-1.0,
       low_z_pivot=0.3,
       high_z_pivot=1.0,
   )

   shear_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
   )

   luminosity_tracer = ccl.WeakLensingTracer(
       cosmology,
       dndz=(z, nz),
       ia_bias=(z, amplitude),
   )

   ell = np.unique(
       np.geomspace(10, 3000, 120).astype(int)
   )

   linear_power = cosmology.get_linear_power()

   cell_gg = ccl.angular_cl(
       cosmology,
       shear_tracer,
       shear_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   cell_luminosity = ccl.angular_cl(
       cosmology,
       luminosity_tracer,
       luminosity_tracer,
       ell,
       p_of_k_a=linear_power,
   )

   prefactor = ell * (ell + 1.0) / (2.0 * np.pi)

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       ell,
       prefactor * cell_gg,
       color=PLIMA_BLUE,
       linewidth=2.5,
       linestyle="--",
       label="No IA",
   )
   ax.plot(
       ell,
       prefactor * cell_luminosity,
       color=PLIMA_RED,
       linewidth=2.8,
       label="Luminosity-dependent LA",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Angular multipole $\ell$")
   ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
   ax.set_title("Luminosity-function-dependent LA model")
   ax.legend(frameon=False)

   fig.tight_layout()