# plima

[![CI](https://img.shields.io/github/actions/workflow/status/cosmology-kit/plima/ci.yml?branch=main&label=CI&color=3b9ab2&style=flat-square)](https://github.com/cosmology-kit/plima/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-main-3b9ab2?style=flat-square)](https://cosmology-kit.github.io/plima/main/)
[![License](https://img.shields.io/badge/license-MIT-3b9ab2?style=flat-square)](LICENSE)

**PLIMA** is the **Python Library for Intrinsic alignment Models and Amplitudes**.

PLIMA provides simple tools for working with intrinsic alignment models in cosmology.
It includes model level amplitude functions and [CCL](https://ccl.readthedocs.io/en/latest/index.html) 
backend helpers for building intrinsic alignment inputs used in weak lensing analyses.

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
pip install -e ".[dev]"
```


The development documentation is available at:

https://cosmology-kit.github.io/plima/main/

## Citation

If you use PLIMA in your research, please cite it.

```bibtex
@software{sarcevic2026plima,
  title   = {PLIMA: Python Library for Intrinsic alignment Models and Amplitudes},
  author  = {Šarčević, Nikolina},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/cosmology-kit/plima}
}
```

## License

MIT License © 2026 Nikolina Šarčević and contributors.