Halo-model intrinsic alignments
===============================

Overview
--------

Halo models describe large-scale structure by assigning matter and galaxies to
dark-matter halos. intrinsic alignment correlations can then be separated into
contributions from galaxies in the same halo and galaxies in different halos.

A generic intrinsic alignment spectrum may be written as

.. math::

   P_{XY}^{\mathrm{IA}}(k,z)
   =
   P_{XY}^{1\mathrm{h}}(k,z)
   +
   P_{XY}^{2\mathrm{h}}(k,z),

where :math:`X` and :math:`Y` may represent matter or intrinsic galaxy shape.

The two-halo term describes correlations between objects in different halos.
On sufficiently large scales, it is commonly connected to a tidal-alignment
or linear-alignment response.

The one-halo term describes alignments between galaxies and matter within the
same halo. It is particularly relevant on small and intermediate scales, where
central and satellite galaxy orientations depend on the internal structure of
their host halo.

One-halo and two-halo contributions
-----------------------------------

Two-halo contribution
~~~~~~~~~~~~~~~~~~~~~

The two-halo term captures correlations between separate halos,

.. math::

   P_{XY}^{2\mathrm{h}}(k,z).

For intrinsic alignments, its amplitude is controlled in PLIMA by the
large-scale parameter :math:`A_{\mathrm{IA}}(z)`.

One-halo contribution
~~~~~~~~~~~~~~~~~~~~~

The one-halo term captures correlations within a single halo,

.. math::

   P_{XY}^{1\mathrm{h}}(k,z).

A common physical picture places an aligned central galaxy near the halo
center and allows satellite shapes to preferentially point toward or away from
the halo center.

PLIMA parameterizes the amplitude of this satellite contribution using
:math:`A_{1\mathrm{h}}(z)`. Its radial behavior is represented schematically
by a power law,

.. math::

   \gamma_{\mathrm{sat}}^{I}(r,z)
   \propto
   A_{1\mathrm{h}}(z)\,r^b,

where :math:`b` is the satellite-shear radial slope.

The precise normalization, halo profile, and Fourier-space transformation are
defined by the downstream halo-model implementation.

Redshift-dependent parameters
-----------------------------

PLIMA defines the large-scale IA amplitude as

.. math::

   A_{\mathrm{IA}}(z)
   =
   a_{\mathrm{IA}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{\mathrm{IA}}}.

Here

- :math:`a_{\mathrm{IA}}` is the large-scale amplitude at the pivot redshift,
  and
- :math:`\eta_{\mathrm{IA}}` controls its redshift evolution.

The satellite one-halo amplitude is

.. math::

   A_{1\mathrm{h}}(z)
   =
   a_{1\mathrm{h}}
   \left(
      \frac{1+z}{1+z_{\mathrm{pivot}}}
   \right)^{\eta_{1\mathrm{h}}},

where

- :math:`a_{1\mathrm{h}}` is the satellite-alignment amplitude at the pivot
  redshift, and
- :math:`\eta_{1\mathrm{h}}` controls its redshift evolution.

The radial slope is constant on the supplied redshift grid,

.. math::

   b(z) = b.

Transition and damping scales
-----------------------------

PLIMA optionally accepts two scale parameters:

``k_1h``
   A one-halo transition scale.

``k_2h``
   A two-halo damping scale.

When supplied, each is returned as a constant array on the requested redshift
grid,

.. math::

   k_{1\mathrm{h}}(z) = k_{1\mathrm{h}},

and

.. math::

   k_{2\mathrm{h}}(z) = k_{2\mathrm{h}}.

These parameters can be used by a downstream model to control how the one-halo
and two-halo terms enter or transition across physical scales.

PLIMA does not impose a particular transition function. The interpretation of
``k_1h`` and ``k_2h`` belongs to the backend that constructs the final
power spectra.

Model parameters
----------------

The halo-model IA helper uses the following parameters:

``a_ia``
   Large-scale intrinsic alignment amplitude at the pivot redshift.

``eta_ia``
   Redshift-evolution index of the large-scale amplitude.

``a1h``
   Satellite one-halo alignment amplitude at the pivot redshift.

``eta_1h``
   Redshift-evolution index of the one-halo amplitude.

``b``
   Satellite-shear radial power-law slope.

``z_pivot``
   Pivot redshift used for both amplitude-evolution terms.

``k_1h``
   Optional one-halo transition scale.

``k_2h``
   Optional two-halo damping scale.

PLIMA implementation
--------------------

The function

:func:`plima.models.halo_model.halo_model_ia_parameters`

returns the redshift-dependent parameter arrays required by a downstream
halo-model calculation.

The returned dictionary has the form

.. code-block:: text

   {
       "a_ia": a_ia_of_z,
       "a1h": a1h_of_z,
       "b": b_of_z,
   }

If the optional scales are supplied, it additionally contains

.. code-block:: text

   {
       "k_1h": k_1h_of_z,
       "k_2h": k_2h_of_z,
   }

This function does not construct one-halo or two-halo power spectra. It only
prepares and validates the phenomenological parameter evolution.

The final halo-model prediction must specify additional ingredients such as

- a halo mass function,
- halo bias,
- central and satellite occupation,
- halo density profiles,
- galaxy-shape profiles, and
- the prescription used to combine the one-halo and two-halo terms.

References
----------

- Schneider and Bridle (2010), *A halo model for intrinsic alignments of
  galaxy ellipticities*, arXiv:0903.3870.
- Cooray and Sheth (2002), halo-model review, arXiv:astro-ph/0206508.