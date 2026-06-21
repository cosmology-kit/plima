# plima

[![CI](https://img.shields.io/github/actions/workflow/status/cosmology-kit/plima/ci.yml?branch=main\&label=CI\&color=3b9ab2\&style=flat-square)](https://github.com/cosmology-kit/plima/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/cosmology-kit/plima?color=3b9ab2\&style=flat-square)](LICENSE)

**PLIMA** is the **Python Library for Intrinsic Alignment Models and Amplitudes**.

PLIMA provides lightweight tools for working with intrinsic alignment models in cosmology. It includes model level amplitude functions and CCL backend helpers for building intrinsic alignment inputs used in weak lensing calculations.

## Models

PLIMA currently includes:

* LA
* NLA
* TATT
* halo model IA

## Installation

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/cosmology-kit/plima.git
cd plima
pip install -e ".[test]"
```

## Running tests

```bash
pytest -q
```

## Citation

If you use PLIMA in your research, please cite the software using the metadata in `CITATION.cff`.

## License

MIT License © 2026 Nikolina Šarčević and contributors.
