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
    ["py:class", r"arraycontext.typing._UserDefinedArrayContainer"],
    ["py:class", r"arraycontext.typing._UserDefinedArithArrayContainer"],
    ["py:class", r"T"],
]


sphinxconfig_missing_reference_aliases = {
    # numpy
    "NDArray": "obj:numpy.typing.NDArray",
    # pytools
    "ObjectArrayND": "obj:pytools.obj_array.ObjectArrayND",
    # pyopencl
    "WaitList": "obj:pyopencl.WaitList",
    # pymbolic
    "Variable": "class:pymbolic.primitives.Variable",
    "Expression": "obj:pymbolic.typing.Expression",
    "ArithmeticExpression": "obj:pymbolic.ArithmeticExpression",
    "MultiVector": "obj:pymbolic.geometric_algebra.MultiVector",
    # arraycontext
    "ScalarLike": "obj:arraycontext.ScalarLike",
    "ArrayOrContainerOrScalar": "obj:arraycontext.ArrayOrContainerOrScalar",
    "PyOpenCLArrayContext": "class:arraycontext.PyOpenCLArrayContext",
    # modepy
    "mp.Shape": "class:modepy.Shape",
    # meshmode
    "Discretization": "class:meshmode.discretization.Discretization",
    "DOFArray": "class:meshmode.dof_array.DOFArray",
    # sumpy
    "Kernel": "class:sumpy.kernel.Kernel",
    "P2PBase": "class:sumpy.p2p.P2PBase",
    "ExpansionFactoryBase": "class:sumpy.expansion.ExpansionFactoryBase",
    # pytential
    "sym.IntG": "class:pytential.symbolic.primitives.IntG",
    "sym.DOFDescriptor": "class:pytential.symbolic.dof_desc.DOFDescriptor",
    "Operand": "obj:pytential.symbolic.primitives.Operand",
    "QBXForcedLimit": "obj:pytential.symbolic.primitives.QBXForcedLimit",
    "TargetOrDiscretization": "obj:pytential.target.TargetOrDiscretization",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
