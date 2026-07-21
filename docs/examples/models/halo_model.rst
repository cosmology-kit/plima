Halo model examples
===================

The halo-model IA helper prepares redshift-dependent parameters for a
downstream one-halo and two-halo calculation.

The large-scale amplitude is

.. math::

   A_{\mathrm{IA}}(z)
   =
   a_{\mathrm{IA}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{\mathrm{IA}}},

while the satellite one-halo amplitude is

.. math::

   A_{1\mathrm{h}}(z)
   =
   a_{1\mathrm{h}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{1\mathrm{h}}}.

PLIMA also accepts a satellite radial slope and optional one-halo and two-halo
scale parameters.

Redshift-dependent halo-model amplitudes
----------------------------------------

The following example evaluates the large-scale and satellite one-halo
amplitudes.

.. plot::
   :include-source:
   :caption: Large-scale and one-halo IA amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   z = np.linspace(0.0, 3.0, 300)

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=0.8,
       a1h=0.05,
       eta_1h=-0.5,
       b=-2.0,
       z_pivot=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       z,
       parameters["a_ia"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$A_{\rm IA}(z)$",
   )
   ax.plot(
       z,
       parameters["a1h"],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$A_{1{\rm h}}(z)$",
   )

   ax.axvline(
       0.62,
       color="0.5",
       linestyle="--",
       linewidth=1.5,
       label=r"$z_{\rm pivot}$",
   )
   ax.set_yscale("log")
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Amplitude")
   ax.set_title("Halo-model IA amplitudes")
   ax.legend(frameon=False)

   fig.tight_layout()

Effect of redshift-evolution parameters
---------------------------------------

Positive and negative evolution indices produce different redshift trends.
This example varies :math:`\eta_{\mathrm{IA}}` while keeping the pivot
amplitude fixed.

.. plot::
   :include-source:
   :caption: Dependence of the large-scale IA amplitude on its redshift-evolution index.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   COLORS = ["#3B9AB2", "#EBCC2A", "#F21A00"]
   ETA_VALUES = [-1.0, 0.0, 1.0]

   z = np.linspace(0.0, 3.0, 300)

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   for eta_ia, color in zip(ETA_VALUES, COLORS, strict=True):
       parameters = halo_model_ia_parameters(
           z,
           a_ia=1.0,
           eta_ia=eta_ia,
           a1h=0.05,
           eta_1h=0.0,
           b=-2.0,
           z_pivot=0.62,
       )

       ax.plot(
           z,
           parameters["a_ia"],
           color=color,
           linewidth=2.5,
           label=rf"$\eta_{{\rm IA}}={eta_ia:.1f}$",
       )

   ax.axvline(
       0.62,
       color="0.5",
       linestyle="--",
       linewidth=1.5,
       label=r"$z_{\rm pivot}$",
   )
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"$A_{\rm IA}(z)$")
   ax.set_title("Large-scale amplitude evolution")
   ax.legend(frameon=False)

   fig.tight_layout()

Optional scale parameters
-------------------------

The optional ``k_1h`` and ``k_2h`` arguments are returned as constant arrays on
the supplied redshift grid.

The PLIMA helper does not prescribe how these scales enter the final power
spectrum. Their interpretation belongs to the downstream halo-model backend.

.. plot::
   :include-source:
   :caption: Optional one-halo transition and two-halo damping scales.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   z = np.linspace(0.0, 3.0, 300)

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=0.5,
       a1h=0.05,
       eta_1h=-0.5,
       b=-2.0,
       z_pivot=0.62,
       k_1h=1.0,
       k_2h=0.3,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       z,
       parameters["k_1h"],
       color=PLIMA_RED,
       linewidth=2.5,
       label=r"$k_{1{\rm h}}$",
   )
   ax.plot(
       z,
       parameters["k_2h"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$k_{2{\rm h}}$",
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Scale parameter")
   ax.set_title("Halo-model scale parameters")
   ax.legend(frameon=False)

   fig.tight_layout()

Inspecting the returned parameters
----------------------------------

All returned values are NumPy ``float64`` arrays evaluated on the supplied
redshift grid.

.. code-block:: python

   import numpy as np

   from plima.models.halo_model import halo_model_ia_parameters


   z = np.linspace(0.0, 2.0, 100)

   parameters = halo_model_ia_parameters(
       z,
       a_ia=1.0,
       eta_ia=0.5,
       a1h=0.05,
       eta_1h=-0.5,
       b=-2.0,
       z_pivot=0.62,
       k_1h=1.0,
       k_2h=0.3,
   )

   print(parameters.keys())

This prints

.. code-block:: text

   dict_keys(['a_ia', 'a1h', 'b', 'k_1h', 'k_2h'])

The function prepares only the phenomenological parameter evolution. A complete
halo-model prediction additionally requires ingredients such as a halo mass
function, halo bias, occupation model, matter profiles, and intrinsic-shape
profiles.