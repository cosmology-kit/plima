.. |plimalogo| image:: /assets/logos/logo-plima.png
   :alt: PLIMA logo black
   :width: 35px

|plimalogo| Contributing
========================

Contributions in any shape or form are appreciated.
Below are some minimal guidelines to get started.

Development of ``plima`` is organised around the
`GitHub repository <https://github.com/cosmology-kit/plima>`__.
Contributing usually requires
`setting up an account <https://github.com/signup>`__
on GitHub.
No worries, it is free of charge!

When submitting contributions, please write in clear and correct English using
full sentences.
Be concise and avoid unnecessarily formulaic descriptions.


Submitting bugs or code
-----------------------

Submitting a bug report or feature request can be done by
`opening an issue on GitHub <https://github.com/cosmology-kit/plima/issues/new/choose>`__.

In the case of a bug report, please make sure to

* describe the expected behaviour,
* describe the actual behaviour,
* specify the version or versions of ``plima`` that produce the bug,
* include any relevant environment details.

In the case of a feature request, please make sure to

* describe the proposed feature,
* describe the need for the feature.

Submitting a code contribution can be done by
`opening a pull request on GitHub <https://github.com/cosmology-kit/plima/compare>`__.

In the pull request, please make sure of the following:

* The pull request description contains a high-level overview of what is
  implemented in the contribution.
* The description explains the reason for the addition.
  Specifically, it describes the problem the contribution is intended to
  solve.
  If the pull request resolves a bug or implements a feature for which an
  issue exists, make sure to refer to that issue.
* The ``plima`` workflows complete successfully.
  The workflows will run when a pull request is created, but they can also be
  run locally as described below.


Running ``plima`` workflows
---------------------------

``plima`` uses `tox <https://tox.wiki>`__ to run its workflows.
It can be installed along with ``plima`` by adding the ``dev`` dependency
group::

   pip install --group dev plima

All workflows can be run consecutively by calling tox from the project root
directory::

   tox

Specific workflows can also be run in isolation.
The following workflows are provided.


Linting
^^^^^^^

Code for ``plima`` must comply with
`PEP 8 <https://peps.python.org/pep-0008/>`__.
Comments and docstrings must be compatible with
`Google-style comments and docstrings <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`__.

The linting workflow can be run using::

   tox -e lint

Tox uses `Ruff <https://docs.astral.sh/ruff/>`__ as the underlying linter.
Options can be passed to ``ruff check`` by supplying them as command-line
arguments to tox.
For example, to address automatically fixable linting errors, use::

   tox -e lint -- --fix


Documentation
^^^^^^^^^^^^^

Documentation is written in
`reStructuredText <https://docutils.sourceforge.io/rst.html>`__.
API documentation is generated from docstrings, and the complete documentation
can be built using::

   tox -e docs

The generated documentation will be placed in the ``docs/_build`` directory.
Newly created reStructuredText files may need to be added manually to the
appropriate table-of-contents files.

Note that ``tox -e docs`` uses the ``html`` builder of ``sphinx-build``.
A different builder can be selected by passing it as a command-line argument
to tox.
For example, to run the doctest builder::

   tox -e docs -- doctest

A list of supported options can be found on the Sphinx
`Builders <https://www.sphinx-doc.org/en/master/usage/builders/index.html#builders>`__
page.

The complete documentation for the head of the main branch and all release
tags can be generated using::

   tox -e docs-releases


Testing
^^^^^^^

Contributions that contain new code must include tests in the appropriate
files in the ``tests`` directory.
The test suite can be run locally using::

   tox -m test

This attempts to run the test suite for all supported Python versions.

To run the test suite for a specific Python version, call the corresponding
tox environment.
For example, to test against Python 3.13::

   tox -e py313

During development, it is sometimes useful to run a single test file because
this is much faster than running the entire test suite.
To do so, pass the file path as a command-line argument separated from the tox
invocation by ``--``.
For example, to run only the tests in ``tests/test_nla.py`` for all supported
Python versions::

   tox -m test -- tests/test_nla.py