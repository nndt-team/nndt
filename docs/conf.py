import os
import sys

project = "NNDT"
copyright = ""
author = "Konstantin Ushenin"  # 'NNDT-team'

master_doc = "index"  #

extensions = [
    "nbsphinx",  # provides a source parser for .ipynb files
    "sphinx.ext.autodoc",  # include docs from docstrings
    "sphinx.ext.mathjax",  # render math via JavaScript
    "sphinx.ext.autosummary",  # generate autodoc summaries
    "sphinx.ext.napoleon",  # support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # add links to highlighted source code
    "sphinx_copybutton",  # add a "copy" button to code blocks.
    "sphinx_tabs.tabs",  # create tabbed content in Sphinx documentation
]

templates_path = ["_templates"]

# todo_include_todos = True  # as GPJax

nb_execution_mode = "auto"  # Execution mode for notebooks. 'off', 'force', 'cache', 'inline' are available
nbsphinx_allow_errors = (
    False  # stop build process if an exception is raised (in .ipynb)
)
# nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}"]
nbsphinx_responsive_width = "700px"

# LaTex config example:
# https://github.com/JaxGaussianProcesses/GPJax/blob/master/docs/conf.py#LL122

# Variables to decorate cite
# html_static_path = ["_static"]
# html_css_files = ["css/nndt_theme.css"]

autosummary_generate = True
autodoc_typehints = "none"  # As GPJax. "signature", "description", "none", "both"

napoleon_use_rtype = False

html_theme = "sphinx_book_theme"  # https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# Sometimes in the future...
# html_logo = "_static/nndt_logo.svg"
# html_favicon = "_static/nndt_logo.svg"
