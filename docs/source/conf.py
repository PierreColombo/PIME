# Configuration file for the Sphinx documentation builder.

# -- Project information
import sys
import os
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

sys.path.append(os.path.abspath(
    os.path.join(__file__, "../../pimms")
))

sys.path.append(os.path.join(__file__, "../../pimms/continuous_gaussian"))
sys.path.append(os.path.join(__file__, "../../pimms/divergence"))
sys.path.append(os.path.join(__file__, "../../pimms/entropy"))
sys.path.append(os.path.join(__file__, "../../pimms/misc"))
sys.path.append(os.path.join(__file__, "../../pimms/mutual_information"))
sys.path.append(os.path.join(__file__, "../../pimms/unit_tests"))
sys.path.append(os.path.join(__file__, "../pimms/continuous_gaussian"))
sys.path.append(os.path.join(__file__, "../pimms/divergence"))
sys.path.append(os.path.join(__file__, "../pimms/entropy"))
sys.path.append(os.path.join(__file__, "../pimms/misc"))
sys.path.append(os.path.join(__file__, "../pimms/mutual_information"))
sys.path.append(os.path.join(__file__, "../pimms/unit_tests"))
project = 'PIM'
copyright = '2022, Pierre COLOMBO'
author = 'Pierre Colombo, Malik Boudiaf'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autosummary_generate = True
todo_include_todos = True
napoleon_numpy_docstring = True
source_suffix = ['.rst', '.md']

bibtex_bibfiles = [os.path.abspath(
    os.path.join(__file__, "pim")
)]
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
