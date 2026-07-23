"""
This is the configuration file for the project documentation.
You can find the documentation on how to configure Sphinx here:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

docs_root = Path(__file__).parent.parent.resolve()
repo_root = docs_root.parent

# add the project root to the path, so we can load meta-data from __about__
sys.path.insert(0, str(repo_root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from secret_loyalties.__about__ import __authors__, __version__

# format author information
__authors_plain = "; ".join(author["name"] for author in __authors__)
__authors_latex = r"\and ".join(author["name"] for author in __authors__)

project = "secret_loyalties"
author = __authors_plain
copyright = f"Copyright 2025 {__authors_plain}"
release = __version__

# export these variables for usage in rst files
rst_epilog = f"""
.. |project| replace:: {project}
.. |release| replace:: {release}
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.cairosvgconverter",
    "sphinxcontrib.relativeinclude",
]

templates_path = ["templates"]
exclude_patterns = []

language = "en"

# -- Options for output ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

html_theme = "alabaster"
html_static_path = ["static"]

latex_documents = [
    ("index", f"{project}.tex", "", __authors_latex, "manual"),
]

latex_elements = {
    "extraclassoptions": "openany,oneside",
    "papersize": "a4paper",
    "pointsize": "12pt",
    "figure_align": "H",
    "preamble": r"""
        % One line per author on title page
        % cf. https://github.com/jfbu/matplotlib/commit/da0f35a535183d3cf5611063abfe254f0c2be975
        \DeclareRobustCommand{\and}%
          {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
    """,
}


# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

add_module_names = False
autodoc_member_order = "bysource"
