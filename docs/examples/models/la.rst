LA model examples
===================

This page demonstrates the linear alignment (LA) amplitude prescriptions and
the corresponding three-dimensional intrinsic-alignment power spectra.

The LA model multiplies the linear matter power spectrum by the response

.. math::

   F_{\mathrm{IA}}(z)
   =
   -A_{\mathrm{IA}}(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D(z)}.

The matter--intrinsic and intrinsic--intrinsic power spectra are

.. math::

   P_{\delta I}^{\mathrm{LA}}(k,z)
   =
   F_{\mathrm{IA}}(z)P_{\delta}^{\mathrm{lin}}(k,z),

and

.. math::

   P_{II}^{\mathrm{LA}}(k,z)
   =
   F_{\mathrm{IA}}^2(z)P_{\delta}^{\mathrm{lin}}(k,z).

Constant and redshift-dependent amplitudes
------------------------------------------

The simplest LA model has a constant amplitude. PLIMA also provides the
``la_z`` prescription, which evolves linearly with scale factor,

.. math::

   A_{\mathrm{IA}}(z)
   =
   A_{\mathrm{IA}}
   +
   b_{\mathrm{IA}}
   \left[
      \frac{a(z)}{a(z_{\mathrm{pivot}})} - 1
   \right].

The following example compares these two prescriptions.

.. plot::
   :include-source:
   :caption: Constant and scale-factor-dependent LA amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.la import la_amplitude, la_z_amplitude


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   z = np.linspace(0.0, 2.5, 300)

   amplitude_constant = la_amplitude(
       z,
       a_ia=1.0,
   )

   amplitude_redshift = la_z_amplitude(
       z,
       a_ia=1.0,
       b_ia=1.5,
       pivot_redshift=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       z,
       amplitude_constant,
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"Constant $A_{\rm IA}$",
   )
   ax.plot(
       z,
       amplitude_redshift,
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"Scale-factor-dependent $A_{\rm IA}(z)$",
   )

   ax.axhline(0.0, color="0.6", linewidth=1.0)
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"$A_{\rm IA}(z)$")
   ax.set_title("Linear alignment amplitudes")
   ax.legend(frameon=False)

   fig.tight_layout()

Halo-mass-dependent amplitude
-----------------------------

The mass-dependent LA model is

.. math::

   A_{\mathrm{IA}}(M)
   =
   A_{\mathrm{IA}}\,
   f_{\mathrm{red}}
   \left(
      \frac{M_{\mathrm{h}}}{M_{\mathrm{pivot}}}
   \right)^\beta.

This example compares the amplitude for several red-galaxy fractions.

.. plot::
   :include-source:
   :caption: Halo-mass-dependent LA amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.la import la_mass_amplitude


   COLORS = ["#3B9AB2", "#EBCC2A", "#F21A00"]

   halo_mass = np.logspace(11.0, 15.0, 300)
   red_fractions = [0.25, 0.50, 0.75]

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   for red_fraction, color in zip(red_fractions, COLORS, strict=True):
       amplitude = la_mass_amplitude(
           a_ia=1.0,
           red_fraction=np.full_like(halo_mass, red_fraction),
           halo_mass=halo_mass,
           beta=0.5,
           pivot_halo_mass=1.0e13,
       )

       ax.plot(
           halo_mass,
           amplitude,
           color=color,
           linewidth=2.5,
           label=rf"$f_{{\rm red}}={red_fraction:.2f}$",
       )

   ax.axvline(
       1.0e13,
       color="0.5",
       linestyle="--",
       linewidth=1.5,
       label=r"$M_{\rm pivot}$",
   )
   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Halo mass $M_{\rm h}$")
   ax.set_ylabel(r"$A_{\rm IA}(M_{\rm h})$")
   ax.set_title("Halo-mass-dependent LA amplitude")
   ax.legend(frameon=False)

   fig.tight_layout()

LA power spectra
----------------

The spectrum helpers accept a linear matter power spectrum, growth factor, and
matter density parameter. This makes the LA implementation independent of the
cosmology backend.

The following example uses a smooth synthetic linear matter power spectrum to
illustrate the sign and scale dependence of
:math:`P_{\delta I}^{\mathrm{LA}}` and
:math:`P_{II}^{\mathrm{LA}}`.

.. plot::
   :include-source:
   :caption: Synthetic LA matter--intrinsic and intrinsic--intrinsic power spectra.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.la import p_delta_i_la, p_ii_la


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   k = np.logspace(-3.0, 1.0, 400)

   # Smooth synthetic linear matter power spectrum at one redshift.
   linear_matter_power = (
       2.0e4
       * (k / 0.1) ** 0.96
       / (1.0 + (k / 0.18) ** 2) ** 1.8
   )

   growth_factor = np.full_like(k, 0.72)

   p_delta_i = p_delta_i_la(
       linear_matter_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )

   p_ii = p_ii_la(
       linear_matter_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       k,
       -p_delta_i,
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$-P_{\delta I}^{\rm LA}$",
   )
   ax.plot(
       k,
       p_ii,
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$P_{II}^{\rm LA}$",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Wavenumber $k$")
   ax.set_ylabel("Power spectrum")
   ax.set_title("Linear alignment power spectra")
   ax.legend(frameon=False)

   fig.tight_layout()

The matter--intrinsic spectrum is negative for a positive IA amplitude under
the PLIMA sign convention. Its negative is plotted above so that both spectra
can be displayed on logarithmic axes.
