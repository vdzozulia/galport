# docs/source/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information

project = 'galport'
author = 'Zozulia Viktor'

release = '0.1'
version = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "nbsphinx"]
#nbsphinx_prompt_width = 0 # no prompts in nbsphinx

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints/*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'

html_theme = 'sphinx_rtd_theme'
#html_theme = 'agogo'
html_static_path = ['_static']

#body_max_width = 1200
master_doc = 'index'

