def main():
    from pytential import sym
    from pymbolic import var

    ndomains = 5
    k_values = tuple(
            "k%d" % i
            for i in range(ndomains))

    from pytential.symbolic.pde.scalar import TMDielectric2DBoundaryOperator
    pde_op = TMDielectric2DBoundaryOperator(
            k_vacuum=1,
            interfaces=tuple(
                (0, i, sym.DEFAULT_SOURCE)
                for i in range(ndomains)
                ),
            domain_k_exprs=k_values,
            beta=var("beta"))

    op_unknown_sym = pde_op.make_unknown("unknown")

    from pytential.symbolic.mappers import GraphvizMapper
    gvm = GraphvizMapper()
    gvm(pde_op.operator(op_unknown_sym))
    with open("helmholtz-op.dot", "wt") as outf:
        outf.write(gvm.get_dot_code())


if __name__ == "__main__":
    main()
