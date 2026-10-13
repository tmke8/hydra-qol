"""Configuration file for the Sphinx documentation builder.

For a full list of options see:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------

project = "hydra-qol"
copyright = "2026, Thomas M Kehrenberg"
author = "Thomas M Kehrenberg"
release = _version("hydra-qol")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Only document what is listed in `__all__`, in that order.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
}
# Put type hints in the parameter descriptions instead of the (long) signatures.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_class_signature = "separated"
# Show `hydra_qol.check_configs` instead of `hydra_qol.checking.check_configs`.
add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"{project} {release}"
