# docs/source/conf.py
import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information

project = 'galport'
author = 'Zozulia Viktor'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',           
    'nbsphinx',                        
    'sphinx_copybutton',
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

#MOCK_MODULES = ['agama', 'scipy', 'scipy.optimize', 'scipy.integrate', 'scipy.interpolate', 'scipy.signal']
MOCK_MODULES = ['agama']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

#autodoc_mock_imports = ['agama', 'scipy', 'scipy.optimize', 'scipy.integrate', 'scipy.interpolate', 'scipy.signal']

autodoc_mock_imports = ['agama']

autodoc_typehints = "none"