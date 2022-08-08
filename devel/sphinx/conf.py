# Copyleft (C) 2020-2022, Marek Gagolewski <https://www.gagolewski.com>
# Configuration file for the Sphinx documentation builder.

#import sys
#import os
#import sphinx
#import matplotlib.sphinxext
#import IPython.sphinxext
import sphinx_rtd_theme
import genieclust

# -- Project information -----------------------------------------------------

project = 'genieclust'
copyright = '2018–2022, Marek Gagolewski'
author = 'Marek Gagolewski'
html_title = project
html_short_title = project

version = genieclust.__version__
release = version

print("This is %s %s by %s.\n" % (project, version, author))


github_project_url = "https://github.com/gagolews/genieclust/"
html_baseurl = "https://genieclust.gagolewski.com/"

nitpicky = True
smartquotes = True
today_fmt = "%Y-%m-%dT%H:%M:%S%Z"
highlight_language = "r"



extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinxcontrib.bibtex',
    'numpydoc'
    #'sphinx.ext.viewcode',
    #'sphinx.ext.imgmath',
    # 'sphinx.ext.napoleon',
]


myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "dollarmath",
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

todo_include_todos = True

source_suffix = ['.md', '.rst']

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
    #'display_github': True,
    #'github_url': github_project_url,
    #'style_nav_header_background': '#ff704d',
}

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = False

html_last_updated_fmt = today_fmt
html_static_path = ['_static']
html_css_files = ['css/custom.css']


pygments_style = 'colorful'

bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'alpha'
