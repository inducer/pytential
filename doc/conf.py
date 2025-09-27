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
        "DiscretizationStage": "pytential.symbolic.dof_desc.DiscretizationStage",
        "DOFGranularity": "pytential.symbolic.dof_desc.DOFGranularity",
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
    ["py:class", r"_ProxyNeighborEvaluationResult"],
    ["py:class", r"arraycontext.typing._UserDefinedArithArrayContainer"],
    ["py:class", r"arraycontext.typing._UserDefinedArrayContainer"],
]


sphinxconfig_missing_reference_aliases = {
    # numpy
    "NDArray": "obj:numpy.typing.NDArray",
    "np.integer": "obj:numpy.integer",
    "np.floating": "obj:numpy.floating",
    "np.inexact": "obj:numpy.inexact",
    "np.random.Generator": "class:numpy.random.Generator",
    # pytools
    "ObjectArrayND": "obj:pytools.obj_array.ObjectArrayND",
    "T": "obj:pytools.T",
    "obj_array.ObjectArray1D": "obj:pytools.obj_array.ObjectArray1D",
    "obj_array.ObjectArray2D": "obj:pytools.obj_array.ObjectArray2D",
    # pyopencl
    "WaitList": "obj:pyopencl.WaitList",
    # pymbolic
    "ArithmeticExpression": "obj:pymbolic.ArithmeticExpression",
    "Expression": "obj:pymbolic.typing.Expression",
    "MultiVector": "obj:pymbolic.geometric_algebra.MultiVector",
    "Variable": "class:pymbolic.primitives.Variable",
    # arraycontext
    "ArrayContainer": "obj:arraycontext.ArrayContainer",
    "ArrayOrContainerOrScalar": "obj:arraycontext.ArrayOrContainerOrScalar",
    "ArrayOrContainerT": "obj:arraycontext.ArrayOrContainerT",
    "PyOpenCLArrayContext": "class:arraycontext.PyOpenCLArrayContext",
    "ScalarLike": "obj:arraycontext.ScalarLike",
    # modepy
    "mp.Shape": "class:modepy.Shape",
    # meshmode
    "Discretization": "class:meshmode.discretization.Discretization",
    "DOFArray": "class:meshmode.dof_array.DOFArray",
    # boxtree
    "TreeKind": "obj:boxtree.tree_build.TreeKind",
    "FromSepSmallerCrit": "obj:boxtree.traversal.FromSepSmallerCrit",
    # sumpy
    "ExpansionBase": "class:sumpy.expansion.ExpansionBase",
    "ExpansionFactoryBase": "class:sumpy.expansion.ExpansionFactoryBase",
    "Kernel": "class:sumpy.kernel.Kernel",
    "P2PBase": "class:sumpy.p2p.P2PBase",
    # pytential
    "DOFDescriptorLike": "data:pytential.symbolic.dof_desc.DOFDescriptorLike",
    "DOFGranularity": "data:pytential.symbolic.dof_desc.DOFGranularity",
    "DiscretizationStage": "data:pytential.symbolic.dof_desc.DiscretizationStage",
    "GeometryId": "data:pytential.symbolic.dof_desc.GeometryId",
    "KernelArgumentLike": "obj:pytential.symbolic.primitives.KernelArgumentLike",
    "KernelArgumentMapping": "obj:pytential.symbolic.primitives.KernelArgumentMapping",
    "Operand": "obj:pytential.symbolic.primitives.Operand",
    "PotentialMapper": "obj:pytential.symbolic.pde.scalar.PotentialMapper",
    "QBXForcedLimit": "obj:pytential.symbolic.primitives.QBXForcedLimit",
    "Side": "obj:pytential.symbolic.primitives.Side",
    "TargetOrDiscretization": "obj:pytential.target.TargetOrDiscretization",
    "VectorExpression": "obj:pytential.symbolic.pde.scalar.VectorExpression",
    "pytential.symbolic.dof_desc.DOFDescriptorLike":
        "data:pytential.symbolic.dof_desc.DOFDescriptorLike",
    "sym.DOFDescriptor": "class:pytential.symbolic.dof_desc.DOFDescriptor",
    "sym.IntG": "class:pytential.symbolic.primitives.IntG",
    "sym.var": "obj:pytential.symbolic.primitives.var",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
