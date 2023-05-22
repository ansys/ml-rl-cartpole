import warnings
from datetime import datetime

from ansys_sphinx_theme import ansys_favicon, pyansys_logo_black

# suppress annoying matplotlib bug
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)


# -- Project information -----------------------------------------------------
# Project information
project = "ML-RL-Cartpole-PyAnsys"
copyright = f"(c) {datetime.now().year} ANSYS, Inc. All rights reserved"
author = "ANSYS, Inc."
release = version = '0.1.0'


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    'IPython.sphinxext.ipython_console_highlighting'
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/dev", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pyvista": ("https://docs.pyvista.org/", None),
}

# static path
html_static_path = ["_static"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Copy button customization ---------------------------------------------------
# exclude traditional Python prompts from the copied code
copybutton_prompt_text = r">>> ?|\.\.\. "
copybutton_prompt_is_regexp = True


# -- Sphinx Gallery Options ---------------------------------------------------
# sphinx_gallery_conf = {
#     # convert rst to md for ipynb
#     "pypandoc": True,
#     # path to your examples scripts
#     # "examples_dirs": ["../../examples/"],
#     # path where to save gallery generated examples
#     # "gallery_dirs": ["examples"],
#     # Patter to search for example files
#     "filename_pattern": r"\.py",
#     # Remove the "Download all examples" button from the top level gallery
#     "download_all_examples": False,
#     # Sort gallery example by file name instead of number of lines (default)
#     "within_subsection_order": FileNameSortKey,
#     # directory where function granular galleries are stored
#     "backreferences_dir": None,
#     # Modules for which function level galleries are created.  In
#     "doc_module": "ansys-mapdl-core",
#     "image_scrapers": ("pyvista", "matplotlib"),
#     "ignore_pattern": "flycheck*",
#     "thumbnail_size": (350, 350),
# }


# -- Options for HTML output -------------------------------------------------
html_theme = "ansys_sphinx_theme"
html_logo = pyansys_logo_black
html_short_title = html_title = "ML-RL-Cartpole"
html_theme_options = {
    "github_url": "https://github.com/pyansys/ml-rl-cartpole",
    "show_prev_next": False,
    "show_breadcrumbs": True,
    "additional_breadcrumbs": [
        ("PyAnsys", "https://docs.pyansys.com/"),
        ("PyMAPDL", "https://mapdldocs.pyansys.com/"),
        ("Extended Examples Library", "https://mapdldocs.pyansys.com/user_guide/extended_examples/index.html"),
    ],
}

# Favicon
html_favicon = ansys_favicon

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pymapdldoc"


# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, f"pymapdl-ml-cartpole-documentation-{version}.tex",
     "PyMAPDL Cart-Pole Documentation", author, "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "pymapdl-ml-cartpole", "PyMAPDL Cart-Pole Documentation", [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PyMAPDL Cart-Pole Example",
        "PyMAPDL Cart-Pole Example Documentation",
        author,
        "PyMAPDL Cart-Pole Example",
        "Engineering Software",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]
