import os
from pathlib import Path
import re
import sys
import mplcursors

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
_req_path = Path('../../.doc-requirements.txt')
needs_extensions = {
    'sphinx_gallery.gen_gallery':
    dict(line.split('==') for line in _req_path.read_text().splitlines())[
        'sphinx-gallery']

}

source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'

project = 'mplcursors'
copyright = '2016–present, Antony Lee'
author = 'Antony Lee'

# RTD modifies conf.py, making setuptools_scm mark the version as -dirty.
version = release = re.sub(r'\.dirty$', '', mplcursors.__version__)

language = 'en'

default_role = 'any'

pygments_style = 'sphinx'

todo_include_todos = False

python_use_unqualified_type_names = True

# -- Options for HTML output ----------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/anntzer/mplcursors',
}
html_css_files = ['hide_some_gallery_elements.css']
html_static_path = ['_static']

htmlhelp_basename = 'mplcursors_doc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}
latex_documents = [(
    master_doc,
    'mplcursors.tex',
    'mplcursors Documentation',
    'Antony Lee',
    'manual',
)]

# -- Options for manual page output ---------------------------------------

man_pages = [(
    master_doc,
    'mplcursors',
    'mplcursors Documentation',
    [author],
    1,
)]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [(
    master_doc,
    'mplcursors',
    'mplcursors Documentation',
    author,
    'mplcursors',
    'Interactive data selection cursors for Matplotlib.',
    'Miscellaneous',
)]

# -- Misc. configuration --------------------------------------------------

autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
}

# CustomSortKey cannot be defined *here* because it would be unpicklable as
# this file is exec'd rather than imported.
sys.path.append(".")
from _local_ext import CustomSortKey

os.environ.pop("DISPLAY", None)  # Don't warn about non-GUI when running s-g.

sphinx_gallery_conf = {
    'backreferences_dir': None,
    'examples_dirs': '../../examples',
    'filename_pattern': r'.*\.py',
    'gallery_dirs': 'examples',
    'min_reported_time': 1,
    'within_subsection_order': CustomSortKey,
}
