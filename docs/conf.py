"""Sphinx configuration following the DLS Python Copier template."""

from importlib.metadata import version as package_version

project = "lllm2"
author = "Giles Knap"
copyright = "2026, Giles Knap"
release = package_version(project)
version = release
extensions = ["myst_parser", "sphinx_copybutton", "sphinx_design"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
nitpicky = True
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_theme_options = {
    "logo": {"text": project},
    "github_url": "https://github.com/gilesknap/lllm2",
    "use_edit_page_button": True,
    "navbar_end": ["theme-switcher", "icon-links"],
    "show_toc_level": 2,
}
html_context = {
    "github_user": "gilesknap",
    "github_repo": "lllm2",
    "github_version": "main",
    "doc_path": "docs",
}
