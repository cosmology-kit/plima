PLIMA
=====

.. image:: assets/logos/logo-plima.png
   :alt: PLIMA logo
   :width: 120px
   :align: left

PLIMA (Python Library for Intrinsic alignment Models and Amplitudes)
is a Python package for modeling intrinsic alignments (IA) in cosmology.

IA are correlations between galaxy shapes caused by their formation and
evolution within the large-scale tidal field. They contribute to weak-lensing
measurements and must be modeled when interpreting cosmic shear and related
observables.

PLIMA provides a clear interface for constructing IA models, including
redshift- and luminosity-dependent scaling relations. It separates the
astrophysical IA prescription from the cosmology backend used to calculate
the corresponding power spectra.

The package supports multiple IA models through the CCL backend, with direct
access to additional model options through backend keyword arguments. It also
defines explicit sign and normalization conventions and supports both
scientific analyses and forecasting workflows.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   theory/index
   examples/index
   conventions
   contributing
   cite
   api/index
