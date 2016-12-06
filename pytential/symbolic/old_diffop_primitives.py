
# {{{ differential operators on layer potentials

def grad_S(kernel, arg, dim):
    from pytools.obj_array import log_shape
    arg_shape = log_shape(arg)
    result = np.zeros(arg_shape+(dim,), dtype=object)
    from pytools import indices_in_shape
    for i in indices_in_shape(arg_shape):
        for j in range(dim):
            result[i+(j,)] = IntGdTarget(kernel, arg[i], j)
    return result


def grad_D(kernel, arg, dim):
    from pytools.obj_array import log_shape
    arg_shape = log_shape(arg)
    result = np.zeros(arg_shape+(dim,), dtype=object)
    from pytools import indices_in_shape
    for i in indices_in_shape(arg_shape):
        for j in range(dim):
            result[i+(j,)] = IntGdMixed(kernel, arg[i], j)
    return result


def tangential_surf_grad_source_S(kernel, arg, dim=3):
    from pytools.obj_array import make_obj_array
    return make_obj_array([
        IntGdSource(kernel, arg,
            ds_direction=make_tangent(i, dim, "src"))
        for i in range(dim-1)])


def surf_grad_S(kernel, arg, dim):
    """
    :arg dim: The dimension of the ambient space.
    """

    return project_to_tangential(cse(grad_S(kernel, arg, dim)))


def div_S_volume(kernel, arg):
    return sum(IntGdTarget(kernel, arg_n, n) for n, arg_n in enumerate(arg))


def curl_S_volume(kernel, arg):
    from pytools import levi_civita
    from pytools.obj_array import make_obj_array

    return make_obj_array([
        sum(
            levi_civita((l, m, n)) * IntGdTarget(kernel, arg[n], m)
            for m in range(3) for n in range(3))
        for l in range(3)])


def curl_curl_S_volume(k, arg):
    # By vector identity, this is grad div S volume + k^2 S_k(arg),
    # since S_k(arg) satisfies a Helmholtz equation.

    from pytools.obj_array import make_obj_array

    def swap_min_first(i, j):
        if i < j:
            return i, j
        else:
            return j, i

    return make_obj_array([
        sum(IntGd2Target(k, arg[m], *swap_min_first(m, n)) for m in range(3))
        for n in range(3)]) + k**2*S(k, arg)


def nxcurl_S(kernel, loc, arg):
    """
    :arg loc: one of three values:
      * +1 on the side of the surface toward which
        the normal points ('exterior' of the surface),
      * 0 on the surface, or evaluated at a volume target
      * -1 on the interior of the surface.
    """
    nxcurl_S = np.cross(normal(3), curl_S_volume(kernel, arg))
    assert loc in [-1, 0, 1], "invalid value for 'loc' (%s)" % loc
    return nxcurl_S + loc*(1/2)*arg


def surface_laplacian_S_squared(u, invertibility_scale=0):
    """
    :arg u: The field to which the surface Laplacian is applied.
    """
    # http://wiki.tiker.net/HellsKitchen/SurfaceLaplacian

    Su = cse(S(0, u), "su_from_surflap")

    return (
            - 2*mean_curvature()*Sp(0, Su)
            - ((Spp(0, Su)+Dp(0, Su))-(-1/4*u+Sp(0, Sp(0, u))))
            - invertibility_scale * mean(S(0, Su))*Ones())


def S_surface_laplacian_S(u, dim, invertibility_scale=0, qbx_fix_scale=0):
    """
    :arg u: The field to which the surface Laplacian is applied.
    """

    # This converges, but appears to work quite poorly compared to the above.

    tgrad_Su = cse(
            project_to_tangential(grad_S(0, u, dim)),
            "tgrad_su_from_surflap")

    return (
            - IntGdSource(0, Ones(), ds_direction=real(tgrad_Su))
            - 1j*IntGdSource(0, Ones(), ds_direction=imag(tgrad_Su))
            - invertibility_scale * S(0, Ones()*mean(S(0, u)))
            - qbx_fix_scale * (
                u
                # D+ - D- = identity (but QBX will compute the
                # 'compact part of the identity' -- call that I*)
                - (
                    D(0, u, qbx_forced_limit=+1)
                    - D(0, u, qbx_forced_limit=-1))
                )
                # The above is I - I*, which means only the high-frequency
                # bits of the identity are left.
            )

# }}}


# {{{ geometric operations

def xyz_to_tangential(xyz_vec, which=None):
    d = len(xyz_vec)
    x2l = xyz_to_local_matrix(d)
    return np.dot(x2l[:-1], xyz_vec)


def tangential_to_xyz(tangential_vec, which=None):
    d = len(tangential_vec) + 1
    x2l = xyz_to_local_matrix(d)
    return np.dot(x2l[:-1].T, tangential_vec)


def project_to_tangential(xyz_vec, which=None):
    return tangential_to_xyz(
            cse(xyz_to_tangential(xyz_vec, which), which))


def n_dot(vec, which=None):
    return np.dot(normal(len(vec), which), vec)


def n_cross(vec, which=None):
    nrm = normal(3, which)

    from pytools import levi_civita
    from pytools.obj_array import make_obj_array
    return make_obj_array([
        sum(
            levi_civita((i, j, k)) * nrm[j] * vec[k]
            for j in range(3) for k in range(3))
        for i in range(3)])


def surf_n_cross(tangential_vec):
    assert len(tangential_vec) == 2
    from pytools.obj_array import make_obj_array
    return make_obj_array([-tangential_vec[1], tangential_vec[0]])

# }}}


