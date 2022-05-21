from functools import reduce

from pytential.symbolic.mappers import Collector


class DOFDescriptorCollector(Collector):
    r"""Gathers all the :class:`~pytential.symbolic.dof_desc.DOFDescriptor`\ s
    in an expression.
    """

    def map_ones(self, expr):
        return {expr.dofdesc}

    map_is_shape_class = map_ones
    map_q_weight = map_ones
    map_node_coordinate_component = map_ones

    def map_num_reference_derivative(self, expr):
        return {expr.dofdesc} | self.rec(expr.operand)

    def map_interpolation(self, expr):
        return {expr.from_dd, expr.to_dd} | self.rec(expr.operand)

    def map_node_sum(self, expr):
        return self.rec(expr.operand)

    map_node_max = map_node_sum
    map_node_min = map_node_sum

    def map_elementwise_sum(self, expr):
        return {expr.dofdesc} | self.rec(expr.operand)

    map_elementwise_max = map_elementwise_sum
    map_elementwise_min = map_elementwise_sum

    def map_int_g(self, expr):
        import operator
        return ({expr.source, expr.target}
                | reduce(operator.or_, (self.rec(d) for d in expr.densities))
                | reduce(operator.or_,
                    (self.rec(v) for v in expr.kernel_arguments.values()), set())
                )
