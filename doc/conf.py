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

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://documen.tician.de/boxtree/": None,
    "https://numpy.org/doc/stable/": None,
    "https://scipy.github.io/devdocs/": None,
    "https://documen.tician.de/arraycontext/": None,
    "https://documen.tician.de/meshmode/": None,
    "https://documen.tician.de/modepy/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/pytools/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/sumpy/": None,
    }
