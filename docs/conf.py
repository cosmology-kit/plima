# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# Ensure src/ layout imports work
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"colorspacious\.comparison",
)

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "PLIMA"
author = "Nikolina Šarčević"
copyright = "2026, Nikolina Šarčević"

# Remove default "documentation" suffix in browser title
html_title = "PLIMA Documentation"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_multiversion",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -----------------------------------------------------------------------------
# Autodoc / autosummary
# -----------------------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
}

# -----------------------------------------------------------------------------
# Doctest configuration
# -----------------------------------------------------------------------------
doctest_global_setup = r"""
import numpy as np
np.set_printoptions(precision=12, suppress=True)
"""

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_baseurl = "https://cosmology-kit.github.io/plima/"

html_theme = "furo"
html_permalinks_icon = "<span>#</span>"

html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/cosmology-kit/plima/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#3B9AB2",
        "color-brand-content": "#3B9AB2",
        "color-link": "#3B9AB2",
        "color-link--hover": "#F21A00",
        "color-link--visited": "#3B9AB2",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3B9AB2",
        "color-brand-content": "#3B9AB2",
        "color-link": "#3B9AB2",
        "color-link--hover": "#F21A00",
        "color-link--visited": "#3B9AB2",
    },
}

html_css_files = [
    "custom.css",
]

# -----------------------------------------------------------------------------
# Matplotlib plot directive
# -----------------------------------------------------------------------------
plot_html_show_source_link = False

plot_formats = [("png", 300)]

plot_rcparams = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
}

# -----------------------------------------------------------------------------
# Sphinx multiversion
# -----------------------------------------------------------------------------
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
smv_outputdir_format = "{ref.name}"
smv_site_url = "https://cosmology-kit.github.io/plima/"

# -----------------------------------------------------------------------------
# Copybutton configuration
# -----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False