# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import syncfit
from typing import List
import sys
import os

sys.path.append(os.path.abspath(os.pardir))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../src/syncfit"))  # needed for autodocs

# -- Project information -----------------------------------------------------

project = 'syncfit'
copyright = '2024, Noah Franz, Collin Christy'
author = 'Noah Franz, Collin Christy'

# The full version, including alpha/beta/rc tags
release = syncfit.__version__
version = syncfit.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    "*/.virtual_documents/*",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
# html_logo = "logo/path"
html_title = f"syncfit {version}"
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/alexander-group/syncfit",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

html_baseurl = "https://syncfit.readthedocs.io/en/latest/"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: List[str] = ['_static']

# define the rst epilog to access variables elsewhere
rst_epilog = f"""
.. |version| replace:: v{version}
"""
