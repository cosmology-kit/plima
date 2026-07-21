TATT model examples
===================

The tidal alignment and tidal torquing (TATT) model introduces three
redshift-dependent amplitudes:

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

PLIMA converts these amplitudes into the perturbative coefficients
:math:`c_1`, :math:`c_2`, and :math:`c_\delta`.

Redshift-dependent TATT amplitudes
----------------------------------

The following example evaluates the three amplitudes on a common redshift
grid.

.. plot::
   :include-source:
   :caption: Redshift evolution of the dimensionless TATT amplitudes.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.tatt import tatt_amplitudes


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   z = np.linspace(0.0, 3.0, 300)

   amplitudes = tatt_amplitudes(
       z,
       a1=1.0,
       a2=0.5,
       a1delta=0.3,
       eta1=0.5,
       eta2=-0.5,
       eta1delta=1.0,
       z_pivot=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

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

   ax.axvline(
       0.62,
       color="0.5",
       linestyle="--",
       linewidth=1.5,
       label=r"$z_{\rm pivot}$",
   )
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Dimensionless amplitude")
   ax.set_title("TATT amplitudes")
   ax.legend(frameon=False)

   fig.tight_layout()

Normalized perturbative coefficients
------------------------------------

The dimensionless amplitudes are converted into the coefficients used by a
perturbative IA tracer,

.. math::

   c_1(z)
   =
   -A_1(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D(z)},

.. math::

   c_\delta(z)
   =
   -A_{1\delta}(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D(z)},

and

.. math::

   c_2(z)
   =
   5A_2(z)
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D^2(z)}.

For illustration, this example uses the approximate growth history

.. math::

   D(z) = (1+z)^{-0.8}.

A production analysis should instead supply the growth factor from its
cosmology backend.

.. plot::
   :include-source:
   :caption: Normalized TATT perturbative coefficients.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.tatt import tatt_normalized_coefficients


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"
   PLIMA_YELLOW = "#EBCC2A"

   z = np.linspace(0.0, 3.0, 300)
   growth_factor = (1.0 + z) ** -0.8

   coefficients = tatt_normalized_coefficients(
       z,
       growth_factor=growth_factor,
       omega_m=0.3,
       a1=1.0,
       a2=0.5,
       a1delta=0.3,
       eta1=0.5,
       eta2=-0.5,
       eta1delta=1.0,
       z_pivot=0.62,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

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
       label=r"$c_\delta(z)$",
   )

   ax.axhline(0.0, color="0.6", linewidth=1.0)
   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel("Normalized coefficient")
   ax.set_title("TATT perturbative coefficients")
   ax.legend(frameon=False)

   fig.tight_layout()

Comparing the two quadratic conventions
---------------------------------------

PLIMA supports two normalizations for the quadratic coefficient. The default is

.. math::

   c_2
   =
   5A_2
   \frac{C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}}{D^2},

while the alternative is

.. math::

   c_2
   =
   5A_2
   \frac{
      C_1\rho_{\mathrm{crit}}\Omega_{\mathrm{m}}^2
   }{
      \Omega_{\mathrm{m,fid}}D^2
   }.

.. plot::
   :include-source:
   :caption: Comparison of the supported quadratic TATT normalizations.

   import matplotlib.pyplot as plt
   import numpy as np

   from plima.models.tatt import tatt_normalized_coefficients


   PLIMA_BLUE = "#3B9AB2"
   PLIMA_RED = "#F21A00"

   z = np.linspace(0.0, 3.0, 300)
   growth_factor = (1.0 + z) ** -0.8

   default_coefficients = tatt_normalized_coefficients(
       z,
       growth_factor=growth_factor,
       omega_m=0.32,
       a2=1.0,
       eta2=0.5,
       z_pivot=0.62,
       use_omega_m_squared_for_c2=False,
   )

   alternative_coefficients = tatt_normalized_coefficients(
       z,
       growth_factor=growth_factor,
       omega_m=0.32,
       omega_m_fid=0.30,
       a2=1.0,
       eta2=0.5,
       z_pivot=0.62,
       use_omega_m_squared_for_c2=True,
   )

   fig, ax = plt.subplots(figsize=(7.5, 5.0))

   ax.plot(
       z,
       default_coefficients["c2"],
       color=PLIMA_BLUE,
       linewidth=2.5,
       label=r"$c_2 \propto \Omega_{\rm m}$",
   )
   ax.plot(
       z,
       alternative_coefficients["c2"],
       color=PLIMA_RED,
       linewidth=2.5,
       linestyle="--",
       label=(
           r"$c_2 \propto "
           r"\Omega_{\rm m}^2/\Omega_{\rm m,fid}$"
       ),
   )

   ax.set_xlabel(r"Redshift $z$")
   ax.set_ylabel(r"$c_2(z)$")
   ax.set_title("Quadratic TATT normalization")
   ax.legend(frameon=False)

   fig.tight_layout()

Preparing PT tracer inputs
--------------------------

The :func:`plima.models.tatt.tatt_pt_biases` helper returns the coefficients in
the tuple form commonly expected by perturbation-theory tracers.

.. code-block:: python

   import numpy as np

   from plima.models.tatt import tatt_pt_biases


   z = np.linspace(0.0, 3.0, 100)
   growth_factor = (1.0 + z) ** -0.8

   biases = tatt_pt_biases(
       z,
       growth_factor=growth_factor,
       omega_m=0.3,
       a1=1.0,
       a2=0.5,
       a1delta=0.3,
       eta1=0.5,
       eta2=-0.5,
       eta1delta=1.0,
       z_pivot=0.62,
   )

   z_c1, c1 = biases["c1"]
   z_c2, c2 = biases["c2"]
   z_cdelta, cdelta = biases["cdelta"]

The TATT helpers prepare amplitudes and perturbative coefficients. Construction
of the complete TATT power spectra is delegated to the perturbation-theory
backend.