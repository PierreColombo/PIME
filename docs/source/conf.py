# Configuration file for the Sphinx documentation builder.

# -- Project information
import sys
import os
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

sys.path.append(os.path.abspath(
    os.path.join(__file__, "../../pime")
))

sys.path.append(os.path.join(__file__, "../../pime/continuous_gaussian"))
sys.path.append(os.path.join(__file__, "../../pime/divergence"))
sys.path.append(os.path.join(__file__, "../../pime/entropy"))
sys.path.append(os.path.join(__file__, "../../pime/misc"))
sys.path.append(os.path.join(__file__, "../../pime/mutual_information"))
sys.path.append(os.path.join(__file__, "../../pime/unit_tests"))
sys.path.append(os.path.join(__file__, "../pime/continuous_gaussian"))
sys.path.append(os.path.join(__file__, "../pime/divergence"))
sys.path.append(os.path.join(__file__, "../pime/entropy"))
sys.path.append(os.path.join(__file__, "../pime/misc"))
sys.path.append(os.path.join(__file__, "../pime/mutual_information"))
sys.path.append(os.path.join(__file__, "../pime/unit_tests"))
project = 'PIME'
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
source_suffix = ['.rst', '.md']

bibtex_bibfiles = [
    str(pathlib.Path(__file__).parent.parent / 'bib' / 'IEEEabrv.bib'),
    str(pathlib.Path(__file__).parent.parent / 'bib' / 'lib.bib'),
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
