import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "mpest"
copyright = "2025, Anton Kazancev, Danil Totjmyanin"
author = "Anton Kazancev, Danil Totjmyanin"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = ".rst"
master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
