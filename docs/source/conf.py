# docs/source/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'galport'
author = 'Zozulia Viktor'

release = '0.1'
version = '0.1.0'

# -- General configuration

xtensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',           # Для NumPy-style docstrings
    'nbsphinx',                        # Для Jupyter notebooks
    'sphinx_copybutton',               # Для копирования кода
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
