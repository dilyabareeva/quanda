# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "quanda"
copyright = f"{str(datetime.utcnow().year)}, Dilyara Bareeva, Galip Ümit Yolcu"
author = "Dilyara Bareeva, Galip Ümit Yolcu"
release = "05.10.2024"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.autosummary",
]
source_suffix = [".rst", ".md"]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "quanda_panda_no_bg.png",
    "dark_logo": "quanda_panda_black_bg.png",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-background-primary": "#FFFFFF",
        "color-background-secondary": "#FAFAF2",
        "color-highlight-on-target": "#EDDCFF",
        "color-brand-primary": "#396A11",
    },
    "dark_css_variables": {
        "color-background-secondary": "#1A1C18",
        "color-highlight-on-target": "#3F4A34",
        "color-brand-primary": "#DEEBCE",
    },
}

# -- Extension configuration ------------------------------------------------
autodoc_default_options = {
    "special-members": "__call__, __init__",
}
