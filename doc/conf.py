from importlib import metadata
from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-21, Andreas Kloeckner and contributors"
release = metadata.version("pytential")
version = ".".join(release.split(".")[:2])

autodoc_type_aliases = {
        "GeometryLike": "pytential.collection.GeometryLike",
        "DiscretizationStages": "pytential.symbolic.dof_desc.DiscretizationStages",
        "DOFGranularities": "pytential.symbolic.dof_desc.DOFGranularities",
        "DOFDescriptorLike": "pytential.symbolic.dof_desc.DOFDescriptorLike",
        }

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext", None),
    "boxtree": ("https://documen.tician.de/boxtree", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "modepy": ("https://documen.tician.de/modepy", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sumpy": ("https://documen.tician.de/sumpy", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

nitpick_ignore_regex = [
    # Sphinx started complaining about these in 8.2.1(-ish)
    # -AK, 2025-02-24
    ["py:class", r"TypeAliasForwardRef"],
    ["py:class", r"arraycontext.container._UserDefinedArrayContainer"],
    ["py:class", r"arraycontext.container._UserDefinedArithArrayContainer"],
]
