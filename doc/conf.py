extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
#source_encoding = "utf-8-sig"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pytential"
copyright = "2013-20, Andreas Kloeckner and contributors"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "2013.1"
# The full version, including alpha/beta/rc tags.
release = "2013.1"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ---------------------------------------------------

html_theme = "furo"

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://documen.tician.de/boxtree/": None,
    "https://numpy.org/doc/stable/": None,
    "https://docs.scipy.org/doc/scipy/reference/": None,
    "https://documen.tician.de/arraycontext/": None,
    "https://documen.tician.de/meshmode/": None,
    "https://documen.tician.de/modepy/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/pytools/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/sumpy/": None,
    }

autoclass_content = "class"
