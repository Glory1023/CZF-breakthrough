'''Configuration file for the Sphinx documentation builder.'''
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'czf'
copyright = '2020, CGI Lab'
author = 'CGI Lab'

# -- General configuration ---------------------------------------------------
pygments_style = 'sphinx'
extensions = [
    'breathe',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}
html_static_path = ['_static']

# -- Options for Breathe -----------------------------------------------------
breathe_default_project = 'czf'
breathe_default_members = ('members', 'undoc-members')
