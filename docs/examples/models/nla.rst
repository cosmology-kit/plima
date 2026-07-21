NLA model examples
==================

This page demonstrates the nonlinear alignment (NLA) amplitude prescriptions
and power-spectrum helpers.

The NLA response has the same form as the LA response,

.. math::

   F_{\mathrm{IA}}(z)
   =
   -A_{\mathrm{IA}}(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D(z)}.

The distinction is that NLA uses a supplied matter power spectrum that may
include nonlinear clustering,

.. math::

   P_{\delta I}^{\mathrm{NLA}}(k,z)
   =
   F_{\mathrm{IA}}(z)P_{\delta}(k,z),

and

.. math::

   P_{II}^{\mathrm{NLA}}(k,z)
   =
   F_{\mathrm{IA}}^2(z)P_{\delta}(k,z).

Amplitude evolution
-------------------

The constant and scale-factor-dependent NLA amplitudes can be evaluated
directly on a redshift grid.

.. plot::
   :include-source:
   :caption: Constant and scale-factor-dependent NLA amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.nla import nla_amplitude, nla_z_amplitude


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   z = np.linspace(0.0, 2.5, 300)

   amplitude_constant = nla_amplitude(
       z,
       a_ia=1.0,
   )

   amplitude_redshift = nla_z_amplitude(
       z,
       a_ia=1.0,
       b_ia=-1.5,
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
   ax.set_title("Nonlinear alignment amplitudes")
   ax.legend(frameon=False)

   fig.tight_layout()

Luminosity-function-weighted amplitude
--------------------------------------

The luminosity-function-dependent model combines redshift evolution, the
red-galaxy fraction, and a supplied luminosity-weighted average,

.. math::

   A_{\mathrm{IA}}^{\mathrm{LF}}(z)
   \propto
   A_{\mathrm{IA}}\,
   f_{\mathrm{red}}(z)\,
   \langle w_L(z)\rangle.

In a full analysis, the red fraction and luminosity weight may be calculated
from luminosity functions. Here they are represented by smooth synthetic
curves.

.. plot::
   :include-source:
   :caption: Components of a luminosity-function-weighted NLA amplitude.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.nla import lf_nla_amplitude


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"
   PLIMA_GREEN = "#5EB23B"

   z = np.linspace(0.0, 2.5, 300)

   red_fraction = 0.75 * np.exp(-0.25 * z)
   luminosity_weighted_average = 0.8 + 0.45 * z

   amplitude = lf_nla_amplitude(
       z,
       luminosity_weighted_average,
       a_ia=1.0,
       red_fraction=red_fraction,
       eta_low_z=0.5,
       eta_high_z=-1.0,
       low_z_pivot=0.3,
       high_z_pivot=1.0,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       z,
       red_fraction,
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$f_{\rm red}(z)$",
   )
   ax.plot(
       z,
       luminosity_weighted_average,
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$\langle w_L(z)\rangle$",
   )
   ax.plot(
       z,
       amplitude,
       color=PLIMA_RED,
       linewidth=3.0,
       label=r"$A_{\rm IA}^{\rm LF}(z)$",
   )
   ax.axvline(
       1.0,
       color=PLIMA_GREEN,
       linestyle="--",
       linewidth=1.5,
       label=r"$z_{\rm high}$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Amplitude")
   ax.set_title("Luminosity-function-weighted NLA model")
   ax.legend(frameon=False)

   fig.tight_layout()

Comparing LA-like and nonlinear spectra
---------------------------------------

Because PLIMA accepts the matter power spectrum as an input, the effect of the
NLA replacement can be illustrated by evaluating the same IA response using a
linear and a nonlinear synthetic spectrum.

.. plot::
   :include-source:
   :caption: Effect of replacing the linear matter power spectrum by a nonlinear spectrum.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.nla import p_delta_i_nla, p_ii_nla


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"
   PLIMA_GREEN = "#5EB23B"

   k = np.logspace(-3.0, 1.0, 400)

   linear_power = (
       2.0e4
       * (k / 0.1) ** 0.96
       / (1.0 + (k / 0.18) ** 2) ** 1.8
   )

   # A simple synthetic small-scale nonlinear enhancement.
   nonlinear_boost = 1.0 + 4.0 * (k / 0.7) ** 2 / (1.0 + (k / 0.7) ** 2)
   nonlinear_power = linear_power * nonlinear_boost

   growth_factor = np.full_like(k, 0.72)

   p_delta_i_linear = p_delta_i_nla(
       linear_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )
   p_delta_i_nonlinear = p_delta_i_nla(
       nonlinear_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )

   p_ii_linear = p_ii_nla(
       linear_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )
   p_ii_nonlinear = p_ii_nla(
       nonlinear_power,
       growth_factor,
       omega_m=0.3,
       amplitude=1.0,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       k,
       -p_delta_i_linear,
       color=PLIMA_BLUE,
       linewidth=2.2,
       linestyle="--",
       label=r"$-P_{\delta I}$: linear $P_\delta$",
   )
   ax.plot(
       k,
       -p_delta_i_nonlinear,
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$-P_{\delta I}$: nonlinear $P_\delta$",
   )
   ax.plot(
       k,
       p_ii_linear,
       color=PLIMA_GREEN,
       linewidth=2.2,
       linestyle="--",
       label=r"$P_{II}$: linear $P_\delta$",
   )
   ax.plot(
       k,
       p_ii_nonlinear,
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$P_{II}$: nonlinear $P_\delta$",
   )

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlabel(r"Wavenumber $k$")
   ax.set_ylabel("Power spectrum")
   ax.set_title("Nonlinear alignment power spectra")
   ax.legend(frameon=False, fontsize=11)

   fig.tight_layout()

The two calculations use the same IA response. Their difference comes entirely
from the supplied matter power spectrum.
