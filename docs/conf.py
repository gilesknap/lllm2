"""Sphinx configuration, using the same theme as lllm3090."""

from importlib.metadata import version as package_version

project = 'lllm2'
author = 'Giles Knap'
copyright = '2026, Giles Knap'
release = package_version(project)
version = release
extensions = ['myst_parser', 'sphinx_copybutton']
myst_heading_anchors = 3
nitpicky = True
exclude_patterns = ['_build']
html_theme = 'pydata_sphinx_theme'
html_title = f'{project} {release}'
html_theme_options = {
    'logo': {'text': project},
    'github_url': 'https://github.com/gilesknap/lllm2',
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'show_toc_level': 2,
}
