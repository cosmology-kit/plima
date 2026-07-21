CCL backend: halo-model IA
==========================

This page demonstrates how PLIMA prepares redshift-dependent intrinsic-
alignment parameters for a halo-model calculation.

The current PLIMA halo-model interface prepares the large-scale, one-halo, and
transition parameters. These quantities can then be passed to a downstream
halo-model implementation.

Large-scale and one-halo amplitudes
-----------------------------------

The halo-model parameterization contains a large-scale IA amplitude and a
satellite one-halo amplitude,

.. math::

   A_{\mathrm{IA}}(z)
   =
   A_{\mathrm{IA}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{\mathrm{IA}}},

and

.. math::

   A_{\mathrm{1h}}(z)
   =
   A_{\mathrm{1h}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{\mathrm{1h}}}.

.. plot::
   :include-source:
   :caption: Large-scale and one-halo IA amplitude evolution.

   import matplotlib.pyplot as plt
   import numpy as np
   import pyccl as ccl

   from plima.models.halo_model import halo_model_ia_parameters


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

   z = np.linspace(0.0, 3.0, 400)

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=-0.5,
       a1h=0.002,
       eta_1h=1.0,
       b=-2.0,
       z_pivot=0.62,
       k_1h=1.0,
       k_2h=0.2,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       z,
       parameters["a_ia"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$A_{\rm IA}(z)$",
   )
   ax.plot(
       z,
       parameters["a1h"] / parameters["a1h"][0],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$A_{\rm 1h}(z)/A_{\rm 1h}(0)$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Relative amplitude")
   ax.set_title("Halo-model IA amplitude evolution")
   ax.legend(frameon=False)

   fig.tight_layout()

The one-halo amplitude is divided by its value at zero redshift so that its
redshift evolution can be compared with the order-unity large-scale
amplitude.

Redshift-evolution choices
--------------------------

The two amplitudes have independent redshift-evolution indices.

.. plot::
   :include-source:
   :caption: Dependence of the halo-model IA amplitudes on their evolution indices.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"
   PLIMA_GREEN = "#5EB23B"

   z = np.linspace(0.0, 3.0, 400)

   constant = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=0.0,
       a1h=0.001,
       eta_1h=0.0,
   )

   evolving_large_scale = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=-1.0,
       a1h=0.001,
       eta_1h=0.0,
   )

   evolving_one_halo = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=0.0,
       a1h=0.001,
       eta_1h=1.5,
   )

   both_evolving = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=-1.0,
       a1h=0.001,
       eta_1h=1.5,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       z,
       constant["a_ia"],
       color=PLIMA_BLUE,
       linewidth=2.3,
       label=r"$A_{\rm IA}$: no evolution",
   )
   ax.plot(
       z,
       evolving_large_scale["a_ia"],
       color=PLIMA_RED,
       linewidth=2.3,
       label=r"$A_{\rm IA}$: $\eta_{\rm IA}=-1$",
   )
   ax.plot(
       z,
       evolving_one_halo["a1h"] / evolving_one_halo["a1h"][0],
       color=PLIMA_YELLOW,
       linewidth=2.3,
       label=r"$A_{\rm 1h}/A_{\rm 1h}(0)$: $\eta_{\rm 1h}=1.5$",
   )
   ax.plot(
       z,
       both_evolving["a1h"] / both_evolving["a1h"][0],
       color=PLIMA_GREEN,
       linewidth=2.3,
       linestyle="--",
       label=r"$A_{\rm 1h}/A_{\rm 1h}(0)$: joint model",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Relative amplitude")
   ax.set_title("Halo-model redshift evolution")
   ax.legend(frameon=False, fontsize=10)

   fig.tight_layout()

Scale and slope parameters
--------------------------

The satellite radial slope and transition scales are returned as arrays on the
same redshift grid as the amplitudes.

.. plot::
   :include-source:
   :caption: Halo-model slope and transition-scale parameters.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   z = np.linspace(0.0, 3.0, 400)

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=-0.5,
       a1h=0.002,
       eta_1h=1.0,
       b=-2.0,
       z_pivot=0.62,
       k_1h=1.0,
       k_2h=0.2,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.2))

   ax.plot(
       z,
       -parameters["b"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$-b$",
   )
   ax.plot(
       z,
       parameters["k_1h"],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$k_{\rm 1h}$",
   )
   ax.plot(
       z,
       parameters["k_2h"],
       color=PLIMA_YELLOW,
       linewidth=2.5,
       label=r"$k_{\rm 2h}$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Parameter value")
   ax.set_title("Halo-model transition parameters")
   ax.legend(frameon=False)

   fig.tight_layout()

Preparing parameters for a CCL halo calculation
-----------------------------------------------

The parameter dictionary can be constructed on a redshift grid associated with
a CCL cosmology.

.. code-block:: python

   import numpy as np
   import pyccl as ccl

   from plima.models.halo_model import halo_model_ia_parameters


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

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=-0.5,
       a1h=0.002,
       eta_1h=1.0,
       b=-2.0,
       z_pivot=0.62,
       k_1h=1.0,
       k_2h=0.2,
   )

   large_scale_amplitude = parameters["a_ia"]
   one_halo_amplitude = parameters["a1h"]
   satellite_slope = parameters["b"]
   one_halo_scale = parameters["k_1h"]
   two_halo_scale = parameters["k_2h"]

The current PLIMA helper prepares the halo-model parameter evolution but does
not itself construct a three-dimensional halo-model IA power spectrum.
Consequently, this page plots the complete parameter model rather than
presenting an angular spectrum that is not yet implemented by PLIMA.