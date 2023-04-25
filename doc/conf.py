import os
from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-21, Andreas Kloeckner and contributors"

os.environ["AKPYTHON_EXEC_FROM_WITHIN_WITHIN_SETUP_PY"] = "1"
ver_dic = {}
exec(compile(open("../pytential/version.py").read(),
    "../pytential/version.py", "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

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
    "scipy": ("https://scipy.github.io/devdocs", None),
    "sumpy": ("https://documen.tician.de/sumpy", None),
}
