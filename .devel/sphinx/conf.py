# Copyleft (C) 2020-2023, Marek Gagolewski <https://www.gagolewski.com>
# Configuration file for the Sphinx documentation builder.

import sys, os, sphinx_rtd_theme
sys.path.append(os.getcwd())

import genieclust

pkg_name = "genieclust"
pkg_title = "genieclust"
pkg_version = genieclust.__version__
copyright_year = "2018–2023"
html_baseurl = "https://genieclust.gagolewski.com/"
github_url = "https://github.com/gagolews/genieclust/"
github_star_repo = "gagolews/genieclust"
analytics_id = None  # don't use it! this site does not track its users
author = "Marek Gagolewski"
copyright = f"{copyright_year}"
html_title = f"Python and R Package {pkg_title}"
html_short_title = f"{pkg_title}"
html_favicon = "_static/favicon.png"

html_version_text = f'\
    Python and R Package<br />\
    v{pkg_version}'


pygments_style = 'default'  #'trac' - 'default' is more readable for some
project = f'{pkg_title}'
version = f'by {author}'
release = f'{pkg_version}'

nitpicky = True
smartquotes = True
today_fmt = "%Y-%m-%dT%H:%M:%S%Z"
highlight_language = "python"
html_last_updated_fmt = today_fmt

extensions = [
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    #'sphinxcontrib.proof',  # proof:exercise, proof:example
    #'sphinx_multitoc_numbering',  # so that chapter numbers do not reset across parts [book only]

    # [Python package API docs only]
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'numpydoc'
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "deflist",
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

todo_include_todos = True

source_suffix = ['.md', '.rst']

numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s'
}
numfig_secnum_depth = 1

plot_include_source = True
plot_html_show_source_link = False
plot_pre_code = """
import numpy as np
import genieclust
import matplotlib.pyplot as plt
np.random.seed(123)
"""

doctest_global_setup = plot_pre_code

numpydoc_use_plots = True

# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_imported_members = True
autosummary_generate = True


html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'prev_next_buttons_location': 'both',
    'sticky_navigation': True,
    'display_version': True,
    'style_external_links': True,
    'vcs_pageview_mode': github_star_repo,  # Marek (layout.html)
    'analytics_id': analytics_id,           # Marek (layout.html)
    'display_version': html_version_text    # Marek (layout.html)
}

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = False

html_static_path = ['_static']
html_css_files = ['css/custom.css']


# BibTeX biblography + Marek's custom pybtex style
import alphamarek
bibtex_default_style = "alphamarek"
bibtex_bibfiles = ["bibliography.bib"]
