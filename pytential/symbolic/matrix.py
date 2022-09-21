from __future__ import annotations


__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """
.. autoclass:: MatrixBuilderBase
.. autoclass:: ClusterMatrixBuilderBase
    :show-inheritance:

Full Matrix Builders
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MatrixBuilder
    :show-inheritance:

.. autoclass:: P2PMatrixBuilder
    :show-inheritance:

Cluster Matrix Builders
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QBXClusterMatrixBuilder
    :show-inheritance:

.. autoclass:: P2PClusterMatrixBuilder
    :show-inheritance:
"""

from sys import intern
from typing import TYPE_CHECKING, Any

import numpy as np

import pymbolic.primitives as p
from arraycontext import PyOpenCLArrayContext, flatten, unflatten
from meshmode.dof_array import DOFArray
from pymbolic.geometric_algebra.mapper import (
    EvaluationRewriter as EvaluationRewriterBase,
)
from pytools import memoize_method

from pytential.collection import GeometryCollection
from pytential.linalg.utils import TargetAndSourceClusterList
from pytential.symbolic.primitives import Variable


if TYPE_CHECKING:
    from collections.abc import Sequence

    from meshmode.discretization import Discretization

    import pytential.symbolic.primitives as prim
    from pytential.collection import GeometryCollection
    from pytential.linalg.utils import TargetAndSourceClusterList
    from pytential.symbolic.primitives import Variable


# {{{ helpers

def is_zero(x):
    return isinstance(x, int | float | complex | np.number) and x == 0


def _get_layer_potential_args(actx, places, expr, context=None, include_args=None):
    """
    :arg expr: symbolic layer potential expression containing the kernel arguments.
    :arg include_args: subset of the kernel arguments to evaluate.
    """

    from pytential import bind
    if context is None:
        context = {}

    if include_args is not None:
        kernel_arguments = {
                k: v for k, v in expr.kernel_arguments.items()
                if k in include_args
                }
    else:
        kernel_arguments = expr.kernel_arguments

    from pytential.source import evaluate_kernel_arguments
    return evaluate_kernel_arguments(
            actx,
            lambda expr: bind(places, expr)(actx, **context),
            kernel_arguments, flat=True)

# }}}


# {{{ base classes for matrix builders

class MatrixBuilderBase(EvaluationRewriterBase):
    """
    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 context: dict[str, Any]) -> None:
        """
        :arg dep_expr: symbolic expression for the input block column
            that the builder is evaluating.
        :arg other_dep_exprs: symbolic expressions for the remaining input
            block columns.
        :arg dep_discr: a concrete :class:`~meshmode.discretization.Discretization`
            for the given *dep_expr*.
        :arg places: a :class:`~pytential.collection.GeometryCollection`
            for all the sources and targets the builder is expected to
            encounter.
        """
        super().__init__(context=context)

        self.array_context = actx
        self.dep_expr = dep_expr
        self.other_dep_exprs = other_dep_exprs
        self.dep_discr = dep_discr
        self.places = places

    # {{{

    def get_dep_variable(self):
        return np.eye(self.dep_discr.ndofs, dtype=np.float64)

    def is_kind_vector(self, x):
        return len(x.shape) == 1

    def is_kind_matrix(self, x):
        return len(x.shape) == 2

    # }}}

    # {{{ map_xxx implementation

    def map_variable(self, expr):
        if expr == self.dep_expr:
            return self.get_dep_variable()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super().map_variable(expr)

    def map_subscript(self, expr):
        if expr == self.dep_expr:
            return self.get_dep_variable()
        elif expr in self.other_dep_exprs:
            return 0
        else:
            return super().map_subscript(expr)

    def map_sum(self, expr):
        sum_kind = None

        term_kind_matrix = intern("matrix")
        term_kind_vector = intern("vector")
        term_kind_scalar = intern("scalar")

        result = 0
        for child in expr.children:
            rec_child = self.rec(child)

            if is_zero(rec_child):
                continue

            if isinstance(rec_child, np.ndarray):
                if self.is_kind_matrix(rec_child):
                    term_kind = term_kind_matrix
                elif self.is_kind_vector(rec_child):
                    term_kind = term_kind_vector
                else:
                    raise RuntimeError("unexpected array rank")
            else:
                term_kind = term_kind_scalar

            if sum_kind is None:
                sum_kind = term_kind

            if term_kind != sum_kind:
                raise RuntimeError(
                        f"encountered '{term_kind}' in sum of kind '{sum_kind}'")

            result = result + rec_child

        return result

    def map_product(self, expr):
        mat_result = None
        vecs_and_scalars = 1

        for child in expr.children:
            rec_child = self.rec(child)

            if is_zero(rec_child):
                return 0

            if isinstance(rec_child, np.number | int | float | complex):
                vecs_and_scalars = vecs_and_scalars * rec_child
            elif isinstance(rec_child, np.ndarray):
                if self.is_kind_matrix(rec_child):
                    if mat_result is not None:
                        raise RuntimeError(
                                f"expression is nonlinear in {self.dep_expr}")
                    else:
                        mat_result = rec_child
                else:
                    vecs_and_scalars = vecs_and_scalars * rec_child

        if mat_result is not None:
            if (isinstance(vecs_and_scalars, np.ndarray)
                    and self.is_kind_vector(vecs_and_scalars)):
                vecs_and_scalars = vecs_and_scalars[:, np.newaxis]

            return mat_result * vecs_and_scalars
        else:
            return vecs_and_scalars

    def map_num_reference_derivative(self, expr):
        from pytential import bind, sym
        rec_operand = self.rec(expr.operand)

        assert isinstance(rec_operand, np.ndarray)
        if self.is_kind_matrix(rec_operand):
            raise NotImplementedError("derivatives")

        actx = self.array_context
        dofdesc = expr.dofdesc
        op = sym.num_reference_derivative(
                sym.var("u"),
                expr.ref_axes,
                dofdesc=dofdesc)

        discr = self.places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)

        template_ary = actx.thaw(discr.nodes()[0])
        rec_operand = unflatten(template_ary, actx.from_numpy(rec_operand), actx)

        return actx.to_numpy(flatten(
                bind(self.places, op)(self.array_context, u=rec_operand),
                actx))

    def map_node_coordinate_component(self, expr):
        from pytential import bind, sym
        op = sym.NodeCoordinateComponent(expr.ambient_axis, dofdesc=expr.dofdesc)

        actx = self.array_context
        return actx.to_numpy(flatten(bind(self.places, op)(actx), actx))

    def map_call(self, expr: p.Call):
        arg, = expr.parameters
        rec_arg = self.rec(arg)

        if isinstance(rec_arg, np.ndarray) and self.is_kind_matrix(rec_arg):
            raise RuntimeError("expression is nonlinear in variable")

        assert isinstance(expr.function, p.Variable)
        from numbers import Number
        if isinstance(rec_arg, Number):
            return getattr(np, expr.function.name)(rec_arg)
        else:
            actx = self.array_context

            rec_arg = actx.from_numpy(rec_arg)
            result = getattr(actx.np, expr.function.name)(rec_arg)
            return actx.to_numpy(flatten(result, actx))

    # }}}


class ClusterMatrixBuilderBase(MatrixBuilderBase):
    """Evaluate individual clusters of a matrix operator, as defined by a
    :class:`~pytential.linalg.utils.TargetAndSourceClusterList`.

    Unlike, e.g. :class:`MatrixBuilder`, matrix cluster builders are
    significantly reduced in scope. They are basically just meant
    to evaluate linear combinations of layer potential operators.
    For example, they do not support composition of operators because we
    assume that each operator acts directly on the density.

    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 tgt_src_index: TargetAndSourceClusterList,
                 context: dict[str, Any]) -> None:
        """
        :arg tgt_src_index: a
            :class:`~pytential.linalg.utils.TargetAndSourceClusterList`
            class describing which clusters are going to be evaluated.
        """

        super().__init__(actx, dep_expr, other_dep_exprs, dep_discr, places, context)
        self.tgt_src_index = tgt_src_index

    @property
    @memoize_method
    def _inner_mapper(self):
        # inner_mapper is used to recursively compute the density to
        # a layer potential operator to ensure there is no composition

        return ClusterMatrixBuilderWithoutComposition(self.array_context,
                self.dep_expr,
                self.other_dep_exprs,
                self.dep_discr,
                self.places,
                self.tgt_src_index, self.context)

    def get_dep_variable(self):
        from pytential.linalg.utils import make_index_cluster_cartesian_product
        actx = self.array_context
        tgtindices, srcindices = (
                make_index_cluster_cartesian_product(actx, self.tgt_src_index)
                )

        return np.equal(
                actx.to_numpy(tgtindices), actx.to_numpy(srcindices)
                ).astype(np.float64)

    def is_kind_vector(self, x):
        # NOTE: since matrices are flattened, the only way to differentiate
        # them from a vector is by size
        return x.size == self.tgt_src_index.target.indices.size

    def is_kind_matrix(self, x):
        # NOTE: since matrices are flattened, we recognize them by checking
        # if they have the right size
        return x.size == self.tgt_src_index._flat_total_size


class ClusterMatrixBuilderWithoutComposition(ClusterMatrixBuilderBase):
    def get_dep_variable(self):
        return 1.0

# }}}


# {{{ QBX layer potential matrix builder

# FIXME: PyOpenCL doesn't do all the required matrix math yet.
# We'll cheat and build the matrix on the host.

class MatrixBuilderDirectResamplerCacheKey:
    """Serves as a unique key for the resampler cache in
    :meth:`pytential.collection.GeometryCollection._get_cache`.
    """


class MatrixBuilder(MatrixBuilderBase):
    """A matrix builder that evaluates the full QBX-mediated operator.

    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 context: dict[str, Any],
                 _weighted: bool = True) -> None:
        """
        :arg _weighted: if *True*, the quadrature weights and area elements are
            added to the operator matrix evaluation.
        """
        super().__init__(actx, dep_expr, other_dep_exprs, dep_discr, places, context)
        self.weighted = _weighted

    def map_interpolation(self, expr):
        from pytential import sym

        if expr.to_dd.discr_stage != sym.QBX_SOURCE_QUAD_STAGE2:
            raise RuntimeError("can only interpolate to QBX_SOURCE_QUAD_STAGE2")
        operand = self.rec(expr.operand)
        actx = self.array_context

        if isinstance(operand, int | float | complex | np.number):
            return operand
        elif isinstance(operand, np.ndarray) and operand.ndim == 1:
            conn = self.places.get_connection(expr.from_dd, expr.to_dd)
            discr = self.places.get_discretization(
                    expr.from_dd.geometry, expr.from_dd.discr_stage)
            template_ary = actx.thaw(discr.nodes()[0])

            from pytools import obj_array
            return obj_array.new_1d([
                actx.to_numpy(flatten(
                    conn(unflatten(template_ary, actx.from_numpy(o), actx)),
                    actx))
                for o in operand
                ])
        elif isinstance(operand, np.ndarray) and operand.ndim == 2:
            cache = self.places._get_cache(MatrixBuilderDirectResamplerCacheKey)
            key = (expr.from_dd.geometry,
                    expr.from_dd.discr_stage,
                    expr.to_dd.discr_stage)

            try:
                mat = cache[key]
            except KeyError:
                from meshmode.discretization.connection import (
                    flatten_chained_connection,
                )
                from meshmode.discretization.connection.direct import (
                    make_direct_full_resample_matrix,
                )

                conn = self.places.get_connection(expr.from_dd, expr.to_dd)
                conn = flatten_chained_connection(actx, conn)
                mat = actx.to_numpy(
                    make_direct_full_resample_matrix(actx, conn)
                    )

                # FIXME: the resample matrix is slow to compute and very big
                # to store, so caching it may not be the best idea
                cache[key] = mat

            return mat.dot(operand)
        else:
            raise RuntimeError("unknown operand type: {}".format(type(operand)))

    def map_int_g(self, expr):
        lpot_source = self.places.get_geometry(expr.source.geometry)
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        actx = self.array_context
        assert abs(expr.qbx_forced_limit) > 0

        result = 0
        for kernel, density in zip(expr.source_kernels, expr.densities, strict=True):
            rec_density = self.rec(density)
            if is_zero(rec_density):
                continue

            assert isinstance(rec_density, np.ndarray)
            if not self.is_kind_matrix(rec_density):
                raise NotImplementedError("layer potentials on non-variables")

            # {{{ geometry

            from pytential import bind, sym
            radii = bind(self.places, sym.expansion_radii(
                source_discr.ambient_dim,
                dofdesc=expr.target))(actx)
            centers = bind(self.places, sym.expansion_centers(
                source_discr.ambient_dim,
                expr.qbx_forced_limit,
                dofdesc=expr.target))(actx)

            # }}}

            # {{{ expansion

            local_expn = lpot_source.get_expansion_for_qbx_direct_eval(
                    kernel.get_base_kernel(), (expr.target_kernel,))

            from sumpy.qbx import LayerPotentialMatrixGenerator
            mat_gen = LayerPotentialMatrixGenerator(
                expansion=local_expn, source_kernels=(kernel,),
                target_kernels=(expr.target_kernel,))

            # }}}

            # {{{ evaluate

            kernel_args = _get_layer_potential_args(
                    actx, self.places, expr, context=self.context)

            mat, = mat_gen(actx,
                    targets=flatten(target_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=flatten(source_discr.nodes(), actx, leaf_class=DOFArray),
                    centers=flatten(centers, actx, leaf_class=DOFArray),
                    expansion_radii=flatten(radii, actx),
                    **kernel_args)
            mat = actx.to_numpy(mat)

            if self.weighted:
                waa = bind(self.places, sym.weights_and_area_elements(
                    source_discr.ambient_dim,
                    dofdesc=expr.source))(actx)
                mat[:, :] *= actx.to_numpy(flatten(waa, actx))

            # }}}

            result += mat @ rec_density

        return result

# }}}


# {{{ p2p matrix builder

class P2PMatrixBuilder(MatrixBuilderBase):
    """A matrix builder that evaluates the full point-to-point kernel interactions.

    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 context: dict[str, Any],
                 exclude_self: bool = True,
                 _weighted: bool = False) -> None:
        """
        :arg exclude_self: if *True*, interactions where the source and target
            indices are the same are ignored. This should be *True* in most
            cases where this is possible, since the kernels are singular at those
            points.
        :arg _weighted: if *True*, the quadrature weights and area elements are
            added to the operator matrix evaluation.
        """
        super().__init__(actx, dep_expr, other_dep_exprs, dep_discr, places, context)

        self.exclude_self = exclude_self
        self.weighted = _weighted

    def map_int_g(self, expr):
        source_discr = self.places.get_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_discretization(
                expr.target.geometry, expr.target.discr_stage)

        actx = self.array_context
        target_base_kernel = expr.target_kernel.get_base_kernel()

        result = 0
        for density, kernel in zip(expr.densities, expr.source_kernels, strict=True):
            rec_density = self.rec(density)
            if is_zero(rec_density):
                continue

            assert isinstance(rec_density, np.ndarray)
            if not self.is_kind_matrix(rec_density):
                raise NotImplementedError("layer potentials on non-variables")

            # {{{ generator

            base_kernel = kernel.get_base_kernel()

            from sumpy.p2p import P2PMatrixGenerator
            mat_gen = P2PMatrixGenerator(
                    source_kernels=(base_kernel,),
                    target_kernels=(target_base_kernel,),
                    exclude_self=self.exclude_self)

            # }}}

            # {{{ evaluation

            # {{{ kernel args

            # NOTE: copied from pytential.symbolic.primitives.IntG
            kernel_args = base_kernel.get_args() + base_kernel.get_source_args()
            kernel_args = {arg.loopy_arg.name for arg in kernel_args}

            kernel_args = _get_layer_potential_args(
                    actx, self.places, expr, context=self.context,
                    include_args=kernel_args)

            if self.exclude_self:
                kernel_args["target_to_source"] = actx.from_numpy(
                        np.arange(0, target_discr.ndofs, dtype=np.int64)
                        )

            # }}}

            mat, = mat_gen(actx,
                    targets=flatten(target_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=flatten(source_discr.nodes(), actx, leaf_class=DOFArray),
                    **kernel_args)
            mat = actx.to_numpy(mat)

            if self.weighted:
                from pytential import bind, sym
                waa = bind(self.places, sym.weights_and_area_elements(
                    source_discr.ambient_dim,
                    dofdesc=expr.source))(actx)

                mat[:, :] *= actx.to_numpy(flatten(waa, actx))

            # }}}

            result += mat @ rec_density

        return result

# }}}


# {{{ cluster matrix builders

class QBXClusterMatrixBuilder(ClusterMatrixBuilderBase):
    """A matrix builder that evaluates a cluster (block) using the QBX method.

    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 tgt_src_index: TargetAndSourceClusterList,
                 context: dict[str, Any],
                 exclude_self: bool = False,
                 _weighted: bool = True) -> None:
        """
        :arg exclude_self: this argument is ignored.
        :arg _weighted: if *True*, the quadrature weights and area elements are
            added to the operator matrix evaluation.
        """
        super().__init__(
            actx, dep_expr, other_dep_exprs, dep_discr, places,
            tgt_src_index, context)

        self.weighted = _weighted

    def map_int_g(self, expr):
        lpot_source = self.places.get_geometry(expr.source.geometry)
        source_discr = self.places.get_source_or_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_discr = self.places.get_target_or_discretization(
                expr.target.geometry, expr.target.discr_stage)

        if self.weighted and source_discr is not target_discr:
            raise NotImplementedError

        actx = self.array_context

        result = 0
        assert abs(expr.qbx_forced_limit) > 0

        for kernel, density in zip(expr.source_kernels, expr.densities, strict=True):
            rec_density = self._inner_mapper.rec(density)
            if is_zero(rec_density):
                continue

            if not np.isscalar(rec_density):
                raise NotImplementedError("layer potentials on non-variables")

            # {{{ geometry

            from pytential.linalg.utils import make_index_cluster_cartesian_product
            tgtindices, srcindices = make_index_cluster_cartesian_product(
                    actx, self.tgt_src_index)

            from pytential import bind, sym
            radii = bind(self.places, sym.expansion_radii(
                source_discr.ambient_dim,
                dofdesc=expr.target))(actx)
            centers = bind(self.places, sym.expansion_centers(
                source_discr.ambient_dim,
                expr.qbx_forced_limit,
                dofdesc=expr.target))(actx)

            # }}}

            # {{{ generator

            local_expn = lpot_source.get_expansion_for_qbx_direct_eval(
                    kernel.get_base_kernel(), (expr.target_kernel,))

            from sumpy.qbx import LayerPotentialMatrixSubsetGenerator
            mat_gen = LayerPotentialMatrixSubsetGenerator(
                local_expn,
                source_kernels=(kernel,),
                target_kernels=(expr.target_kernel,))

            # }}}

            # {{{ evaluate

            kernel_args = _get_layer_potential_args(
                    actx, self.places, expr, context=self.context)

            mat, = mat_gen(actx,
                    targets=flatten(target_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=flatten(source_discr.nodes(), actx, leaf_class=DOFArray),
                    centers=flatten(centers, actx, leaf_class=DOFArray),
                    expansion_radii=flatten(radii, actx),
                    tgtindices=tgtindices,
                    srcindices=srcindices,
                    **kernel_args)

            if self.weighted:
                waa = flatten(
                        bind(self.places,
                            sym.weights_and_area_elements(
                                source_discr.ambient_dim,
                                dofdesc=expr.source))(actx),
                        actx)
                mat *= waa[srcindices]

            # }}}

            result += actx.to_numpy(mat) * rec_density

        return result


class P2PClusterMatrixBuilder(ClusterMatrixBuilderBase):
    """A matrix builder that evaluates a cluster (block) point-to-point.

    .. automethod:: __init__
    """

    def __init__(self,
                 actx: PyOpenCLArrayContext,
                 dep_expr: Variable,
                 other_dep_exprs: Sequence[Variable],
                 dep_discr: Discretization,
                 places: GeometryCollection,
                 tgt_src_index: TargetAndSourceClusterList,
                 context: dict[str, Any],
                 exclude_self: bool = False,
                 _weighted: bool = False) -> None:
        """
        :arg exclude_self: if *True*, interactions where the source and target
            indices are the same are ignored. This should be *True* in most
            cases where this is possible, since the kernels are singular at those
            points.
        :arg _weighted: if *True*, the quadrature weights and area elements are
            added to the operator matrix evaluation.
        """
        super().__init__(
            actx, dep_expr, other_dep_exprs, dep_discr, places,
            tgt_src_index, context)

        self.exclude_self = exclude_self
        self.weighted = _weighted

    def map_int_g(self, expr: prim.IntG):
        source_discr = self.places.get_source_or_discretization(
                expr.source.geometry, expr.source.discr_stage)
        target_or_discr = self.places.get_target_or_discretization(
                expr.target.geometry, expr.target.discr_stage)

        actx = self.array_context
        target_base_kernel = expr.target_kernel.get_base_kernel()

        result = 0
        for kernel, density in zip(expr.source_kernels, expr.densities, strict=True):
            rec_density = self._inner_mapper.rec(density)
            if is_zero(rec_density):
                continue

            if not np.isscalar(rec_density):
                raise NotImplementedError

            # {{{ geometry

            from pytential.linalg.utils import make_index_cluster_cartesian_product
            tgtindices, srcindices = make_index_cluster_cartesian_product(
                    actx, self.tgt_src_index)

            # }}}

            # {{{ generator

            base_kernel = kernel.get_base_kernel()

            from sumpy.p2p import P2PMatrixSubsetGenerator
            mat_gen = P2PMatrixSubsetGenerator(
                    source_kernels=(base_kernel,),
                    target_kernels=(target_base_kernel,),
                    exclude_self=self.exclude_self)

            # }}}

            # {{{ evaluation

            # {{{ kernel args

            # NOTE: copied from pytential.symbolic.primitives.IntG
            kernel_args = [*base_kernel.get_args(), *base_kernel.get_source_args()]
            kernel_args = {arg.loopy_arg.name for arg in kernel_args}

            kernel_args = _get_layer_potential_args(
                    actx, self.places, expr, context=self.context,
                    include_args=kernel_args)

            if self.exclude_self:
                kernel_args["target_to_source"] = actx.from_numpy(
                        np.arange(0, target_or_discr.ndofs, dtype=np.int64)
                        )

            # }}}

            mat, = mat_gen(actx,
                    targets=flatten(target_or_discr.nodes(), actx, leaf_class=DOFArray),
                    sources=flatten(source_discr.nodes(), actx, leaf_class=DOFArray),
                    tgtindices=tgtindices,
                    srcindices=srcindices,
                    **kernel_args)

            from meshmode.discretization import Discretization
            if self.weighted and isinstance(source_discr, Discretization):
                from pytential import bind, sym
                waa = bind(self.places, sym.weights_and_area_elements(
                    source_discr.ambient_dim,
                    dofdesc=expr.source))(actx)
                waa = flatten(waa, actx)

                mat *= waa[srcindices]

            result += actx.to_numpy(mat) * rec_density

        return result

# }}}

# vim: foldmethod=marker
