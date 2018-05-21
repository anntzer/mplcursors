#!/usr/bin/env python3

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]
needs_extensions = {'sphinx_gallery.gen_gallery': '0.1.13'}

source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'

project = 'mplcursors'
copyright = '2016–present, Antony Lee'
author = 'Antony Lee'

# RTD modifies conf.py, making versioneer mark the version as -dirty.
import re
import mplcursors
version = release = re.sub(r'\.dirty$', '', mplcursors.__version__)

language = 'en'

default_role = 'any'

pygments_style = 'sphinx'

todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = 'alabaster'
html_sidebars = {'**': ['about.html', 'navigation.html', 'localtoc.html']}
html_theme_options = {
    'description': 'Interactive data selection cursors for Matplotlib.',
    'github_user': 'anntzer',
    'github_repo': 'mplcursors',
    'github_banner': True,
    'github_button': False}
# html_last_updated_fmt = ''  # bitprophet/alabaster#93

htmlhelp_basename = 'mplcursors_doc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, 'mplcursors.tex', 'mplcursors Documentation',
     'Antony Lee', 'manual'),
]

# -- Options for manual page output ---------------------------------------

man_pages = [
    (master_doc, 'mplcursors', 'mplcursors Documentation',
     [author], 1)
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (master_doc, 'mplcursors', 'mplcursors Documentation',
     author, 'mplcursors', 'Interactive data selection cursors for Matplotlib.',
     'Miscellaneous'),
]

# -- Misc. configuration --------------------------------------------------

autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable', None)}

# CustomSortKey cannot be defined *here* because it would be unpicklable as
# this file is exec'd rather than imported.
import sys; sys.path.append(".")
from _local_ext import CustomSortKey

sphinx_gallery_conf = {
    'backreferences_dir': False,
    'examples_dirs': '../../examples',
    'filename_pattern': '.*\.py',
    'gallery_dirs': 'examples',
    'min_reported_time': 1,
    'within_subsection_order': CustomSortKey,
}
