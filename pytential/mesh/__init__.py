from __future__ import division
import numpy as np
import numpy.linalg as la
from pytools import memoize_method




# {{{ mesh class

class Mesh:
    """
    :ivar nodes:
    :ivar element_node_nrs:
    """
    def __init__(self, nodes, elements, order=1):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.element_node_nrs = np.asarray(elements, dtype=np.intp)
        self.order = order

    def __getinitargs__(self):
        return (self.nodes, self.element_node_nrs, self.order)

    @property
    def dimensions(self):
        return self.nodes.shape[1]

    @memoize_method
    def node_tuples(self):
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam
        return list(gnitstam(self.order, 2))

    @property
    @memoize_method
    def elements(self):
        """Return an *(nelements, 3)* array of indices into :attr:`nodes`
        representing the vertices of each element."""
        return self.element_node_nrs[:, [0, self.order, -1]]

    def transform(self, translate=None, scale=None):
        d = self.dimensions
        A = np.eye(d)
        b = np.zeros((d,))

        if translate is not None:
            b[:] = translate
        if scale is not None:
            A.flat[np.arange(0, d*d, d+1, dtype=np.intp)] = scale

        new_points = np.dot(A, self.nodes.T).T + b
        return Mesh(new_points, self.element_node_nrs, order=self.order)

    def transform_func(self, f):
        new_points = np.array([f(pt) for pt in self.nodes])
        return Mesh(new_points, self.element_node_nrs, order=self.order)

    def __len__(self):
        return len(self.centroids)

    def bounding_box(self):
        return (
                np.min(self.nodes, axis=0),
                np.max(self.nodes, axis=0),
                )

    @property
    def vertex_count(self):
        return len(self.nodes)

    @property
    @memoize_method
    def triangles(self):
        return self.nodes[self.elements]

    @property
    @memoize_method
    def areas(self):
        result = native.triangle_area_vec(self.triangles.T)

        if (result == 0).any():
            from warnings import warn
            warn("found zero-area triangle")

        return result

    @property
    @memoize_method
    def centroids(self):
        return np.sum(self.triangles, axis=1)/3

    @property
    @memoize_method
    def local_bases(self):
        """A local orthonormal basis for each triangle, not aligned with
        any of the triangle's edges.

        axes: dof, axis, basis vector
        """
        result = np.empty((len(self), self.dimensions, self.dimensions),
                order="F")

        from hellskitchen.tools import make_basis_from_normal
        for i, normal in enumerate(self.normals):
            result[i] = make_basis_from_normal(normal)

        return result

    @property
    @memoize_method
    def to_global_maps(self):
        """A local-to-global map for each triangle.

        :return: a tuple *origins,maps* where maps is, by shape,
          a list of d-by-(d-1) matrices.
        """
        origins = self.nodes[self.elements[:,0:1]]
        maps = (
                self.nodes[self.elements[:,1:]]
                - origins).transpose(0, 2, 1)

        d = self.dimensions
        return origins.reshape(-1, d), maps

    @property
    @memoize_method
    def normals(self):
        """axes: dof, axis"""
        return native.triangle_norm_vec(self.triangles.T).T

    @property
    @memoize_method
    def tangents(self):
        """axes: dof, tangent nr, axis"""
        return self.local_bases.transpose(0, 2, 1)[:,:self.dimensions-1,:]

    def generate_edges(self):
        for tri_nr, (va, vb, vc) in enumerate(self.elements):
            yield tri_nr, (va, vb)
            yield tri_nr, (vb, vc)
            yield tri_nr, (vc, va)

    @property
    @memoize_method
    def connectivity(self):
        """Return a mapping associating vertex numbers with neighboring vertices."""

        connectivity = {}

        for tri_nr, (v1, v2) in self.generate_edges():
            for va, vb in [(v1, v2), (v2, v1)]:
                connectivity.setdefault(va, set()).add(v2)

        return connectivity

    @property
    @memoize_method
    def vertex_to_triangle(self):
        """Return a mapping associating vertex numbers with neighboring triangles."""

        result = [[] for i in range(self.vertex_count)]

        for tri_nr, vertices in enumerate(self.elements):
            for vert_nr in vertices:
                result[vert_nr].append(tri_nr)

        return [np.array(tri_nrs, dtype=np.intp) for tri_nrs in result]

    @property
    @memoize_method
    def edge_to_triangles(self):
        result = {}

        for tri_nr, (v1, v2) in self.generate_edges():
            key = frozenset([v1, v2])
            result.setdefault(key, []).append(tri_nr)

        return result

    @property
    @memoize_method
    def triangle_to_neighbor_triangles(self):
        result = [set() for i in range(len(self))]

        e2t = self.edge_to_triangles

        for tri_nr, (v1, v2) in self.generate_edges():
            edge_neighbors = e2t[frozenset([v1, v2])]

            if len(edge_neighbors) == 2:
                tri1, tri2 = edge_neighbors
                result[tri1].add(tri2)
                result[tri2].add(tri1)

        return result

    @property
    @memoize_method
    def vertex_normals(self):
        """Compute area-weighted normals at vertices."""
        result = np.zeros((self.vertex_count, 3))

        areas = self.areas
        normals = self.normals

        for vert_nr, neigh_tri_nrs in enumerate(self.vertex_to_triangle):
            neigh_areas = areas[neigh_tri_nrs]

            normal = np.sum(
                    normals[neigh_tri_nrs]*neigh_areas[:,np.newaxis],
                    axis=0)/np.sum(neigh_areas)
            result[vert_nr] = normal/la.norm(normal)

        return result

# }}}




def mesh_from_gmsh(gmsh_source, options=[]):
    from hellskitchen.mesh.reader import mesh_from_gmsh_geo_str
    return mesh_from_gmsh_geo_str(gmsh_source, 2, other_options=options)




# {{{

def read_tri(tri_file):
    words = tri_file.split()
    n_vertices = int(words[0])
    vertex_end = 2+n_vertices*3
    nodes = (np.array(
        [float(x) for x in words[2:vertex_end]])
        .reshape(-1, 3))
    elements = (np.array([int(x)-1 for x in words[vertex_end:]])
            .reshape(-1, 3))
    return Mesh(nodes, elements)




def write_tri(tri_file, mesh):
    tri_file.write("%d %d\n" % (len(mesh.nodes), len(mesh.elements)))
    for v in mesh.nodes:
        tri_file.write(" ".join(repr(vi) for vi in v)+ "\n")
    for v in mesh.elements:
        tri_file.write(" ".join(repr(vi+1) for vi in v)+ "\n")




def bisect_geometry(geo):
    """
    :return: a triple *(coordinates, triangles, between_vert_map)*, where
      *between_vert_map* maps a :class:`frozenset` of 'old' vertex numbers
      to the vertex number between those vertices.

      The numbers of old points are not changed.
    """
    nodes = geo.nodes
    elements = geo.elements

    between_vert_map = {}
    new_points = []

    def get_between_point(vert_a, vert_b):
        key = frozenset([vert_a, vert_b])
        try:
            return between_vert_map[key]
        except KeyError:
            new_nr = len(nodes)+len(new_points)
            between_vert_map[key] = new_nr
            new_points.append(0.5*(
                nodes[vert_a] 
                + nodes[vert_b]))
            return new_nr

    new_tri_vertices = []
    for va, vb, vc in elements:
        #      a
        #     / \
        #    ab-ca
        #   / \ / \
        #  b---bc--c

        vab = get_between_point(va, vb)
        vbc = get_between_point(vb, vc)
        vca = get_between_point(vc, va)
        new_tri_vertices.append((va, vab, vca))
        new_tri_vertices.append(( vb, vbc, vab))
        new_tri_vertices.append((vc, vca, vbc))
        new_tri_vertices.append((vab, vbc, vca))

    return Mesh(np.vstack((nodes, new_points)), 
            np.array(new_tri_vertices)), between_vert_map




def refine_geometry(geo, trips):
    for i in range(trips):
        geo, _ = bisect_geometry(geo)

    return geo

refine_mesh = refine_geometry






def laplacian_smooth_mesh(geo):
    vc = geo.nodes
    new_vc = np.zeros_like(geo.nodes)
    for vert, others in geo.connectivity.iteritems():
        new_vc[vert] = np.average(
                vc[np.array(list(others), dtype=np.intp)],
                axis=0)

    return Mesh(new_vc, geo.elements)




def refine_smooth_mesh(mesh, trips, smooth_count=2):
    for i in range(trips):
        mesh, _ = bisect_geometry(mesh)
        for j in range(smooth_count):
            mesh = laplacian_smooth_mesh(mesh)

    return mesh




def rejoin_disjoint_vertices(mesh, thresh=1e-15):
    nodes = mesh.nodes

    joint_vertex_num = {}

    from pytools.spatial_btree import SpatialBinaryTreeBucket
    btree = SpatialBinaryTreeBucket(
            np.min(nodes, axis=0)-thresh,
            np.max(nodes, axis=0)+thresh,
            )

    from pytools import ProgressBar
    pb = ProgressBar("btree", len(nodes))
    for i, x in enumerate(nodes):
        pb.progress()
        btree.insert(i, (x-thresh, x+thresh))
    pb.finished()

    pb = ProgressBar("vjoin", len(nodes))
    for i, x in enumerate(nodes):
        pb.progress()
        for j in btree.generate_matches(x):
            y = nodes[j]
            if la.norm(x-y) < thresh:
                i_join_set = joint_vertex_num.get(i)
                j_join_set = joint_vertex_num.get(j)

                if i_join_set is None and j_join_set is None:
                    joint_vertex_num[i] = joint_vertex_num[j] = set([i,j])
                elif i_join_set is not None and j_join_set is None:
                    i_join_set.add(j)
                    joint_vertex_num[j] = i_join_set
                elif i_join_set is None and j_join_set is not None:
                    j_join_set.add(i)
                    joint_vertex_num[i] = j_join_set
                elif i_join_set is not None and j_join_set is not None:
                    new_set = i_join_set | j_join_set
                    for k in new_set:
                        joint_vertex_num[k] = new_set
    pb.finished()

    duplicate_vertices = set(vnum
            for vset in joint_vertex_num.itervalues()
            for vnum in vset)
    nonduplicate_vertices = set(range(len(nodes)))- duplicate_vertices

    new_numbering = {}
    new_vertex_count = 0
    for vset in set(frozenset(s) for s in joint_vertex_num.itervalues()):
        for j in vset:
            new_numbering[j] = new_vertex_count
        new_vertex_count += 1

    for i in nonduplicate_vertices:
        new_numbering[i] = len(new_numbering)
        new_vertex_count += 1

    new_numbering = np.array(
            [new_numbering[i] for i in range(len(nodes))],
            dtype=np.intp)
    old_numbering = np.empty(new_vertex_count, dtype=np.intp)
    old_numbering[new_numbering] = np.arange(len(new_numbering))

    new_element_node_nrs = np.asarray([
            tuple(new_numbering[v] for v in tri)
            for tri in mesh.element_node_nrs],
            dtype=np.intp)

    new_vertex_coordinates = nodes[old_numbering]
    return Mesh(new_vertex_coordinates, new_element_node_nrs, order=mesh.order)




def combine_meshes(*meshes):
    nodes = np.vstack([
        mesh.nodes for mesh in meshes])
    offsets = [0] + list(np.cumsum([
        len(mesh.nodes) for mesh in meshes]))
    elements = np.vstack([
        mesh.elements + offset for mesh, offset in zip(meshes, offsets)])

    return Mesh(nodes, elements)




def flip_orientation(geo, flip_flags):
    new_elements = []

    for flip_flag, el in zip(flip_flags, geo.elements):
        if flip_flag:
            new_elements.append(el[::-1])
        else:
            new_elements.append(el)

    return Mesh(geo.nodes, new_elements)




def warp_to_sphere(geo, blend=1):
    new_vc = (geo.nodes.T
            * np.sum(geo.nodes**2, axis=-1)**-0.5).T

    new_vc = (1-blend)*geo.nodes + blend*new_vc
    return Mesh(new_vc, geo.elements)




def trace_curve_with_centroids(mesh, f, closed, a=0, b=1, dx=1e-3):
    dx = dx*(b-a)

    if closed:
        assert la.norm(f(a)-f(b)) < 1e-12

    starting_pt = f(a)
    from pytools import argmin2
    tri_indices = [argmin2(
        (i, la.norm(centroid-starting_pt)) 
        for i, centroid in enumerate(mesh.centroids))]

    t2nt = mesh.triangle_to_neighbor_triangles

    candidates = None
    x = a
    while x < b:
        if candidates is None:
            candidates = [tri_indices[-1]] + list(t2nt[tri_indices[-1]])

        f_x = f(x)
        next_idx = argmin2(
            (i, la.norm(centroid-f_x)) 
            for i, centroid in enumerate(mesh.centroids[candidates]))

        if next_idx == 0:
            # we're still in the same triangle
            x += dx
        else:
            next_tri = candidates[next_idx]

            tri_indices.append(next_tri)
            candidates = None

    # {{{ close path, if requested

    if closed:
        # find shortest path from end to start, minus start node
        from pytools import a_star
        path = a_star(tri_indices[-1], tri_indices[0], t2nt)[1:]
        tri_indices.extend(path)

        if tri_indices[-1] == tri_indices[0]:
            tri_indices.pop()

    # }}}

    # {{{ remove doubled-up segments

    tri_seen_at = {}
    i = 0
    while i < len(tri_indices):
        this_tri = tri_indices[i]

        if this_tri in tri_seen_at:
            last_at = tri_seen_at[this_tri]

            if not closed:
                for tri in tri_indices[last_at:i]:
                    del tri_seen_at[tri]
                del tri_indices[last_at:i]
                i = last_at
            else:
                # Realize that there are two possible loops here, one through
                # the starting point, and 'direct' one. Compare their lengths,
                # delete the smaller one.

                starting_point_loop_length = (last_at-i) % len(tri_indices)
                direct_loop_length = i-last_at

                if direct_loop_length <= starting_point_loop_length:
                    # Delete the direct one.

                    for tri in tri_indices[last_at:i]:
                        del tri_seen_at[tri]
                    del tri_indices[last_at:i]
                    i = last_at
                else:
                    del tri_indices[i:]
                    del tri_indices[:last_at]
                    # we just connected back to the beginning, we're done
                    break

        tri_seen_at[this_tri] = i
        i += 1

    # }}}

    return tri_indices




def create_spanning_surface(line_integral_rule, intervals=10):
    radius_factors = np.linspace(1, 0, intervals, endpoint=True)

    points = line_integral_rule.points

    # axes: radius index, point nr, xyz
    radii_points = radius_factors[:,np.newaxis, np.newaxis] * points

    triangles = []
    outer_points = radii_points[0]
    outer_point_indices = range(len(outer_points))
    vertices = list(outer_points)

    # Perhaps we should use fewer triangles in the inner rings. The best strategy
    # to do this might be to go from the inside out, adding segments as necessary.
    # The only place where this gets slighly tricky is at the outer boundary, where
    # the number of segments is fixed.

    for radius_index in range(1, intervals):
        inner_points = []
        inner_point_indices = []

        def register_point(pt):
            inner_points.append(pt)
            idx = len(vertices)
            inner_point_indices.append(idx)
            vertices.append(pt)
            return idx

        first_point = radii_points[radius_index, 0]
        inner_this_pt_idx = register_point(first_point)

        outer_idx = 0
        inner_idx = 0
        while outer_idx < len(outer_point_indices):
            next_outer_idx = (outer_idx + 1) % len(outer_point_indices)
            next_inner_idx = (inner_idx + 1) % len(outer_point_indices)

            outer_this_pt_idx = outer_point_indices[outer_idx]
            outer_next_pt_idx = outer_point_indices[next_outer_idx]

            outer_dist = la.norm(
                    vertices[outer_next_pt_idx]
                    - vertices[outer_this_pt_idx])

            if next_inner_idx < len(inner_point_indices):
                inner_next_pt_idx = inner_point_indices[next_inner_idx]
            else:
                inner_next_pt_idx = register_point(radii_points[radius_index, next_inner_idx])

            triangles.append((inner_this_pt_idx, outer_this_pt_idx, outer_next_pt_idx))
            if radius_index < intervals-1:
                triangles.append((inner_this_pt_idx, outer_next_pt_idx, inner_next_pt_idx))

            inner_this_pt_idx = inner_next_pt_idx
            outer_idx += 1
            if radius_index < intervals-1:
                inner_idx += 1

        outer_points = inner_points
        outer_point_indices = inner_point_indices

    return Mesh(vertices, triangles)




def find_element_with_normal(mesh, normal, remaining_el_nrs, threshold=1e-5):
    if remaining_el_nrs is None:
        remaining_el_nrs = set(xrange(mesh.normals.shape[0]))

    for el_i in remaining_el_nrs:
        el_normal = mesh.normals[el_i]
        if la.norm(normal-el_normal) < threshold:
            return el_i

    raise RuntimeError("no element found")





def find_flips(mesh, start_el_nr=None, flip_flags=None, remaining_els=None, thresh=1e-5):
    if start_el_nr is None:
        start_el_nr = 0

    if flip_flags is None:
        flip_flags = np.zeros(len(mesh), dtype=np.bool)

    if remaining_els is None:
        remaining_els = set(xrange(len(mesh)))
        remaining_els.remove(start_el_nr)

    t2t = mesh.triangle_to_neighbor_triangles

    queue = [ # pairs of (from, to)
            (start_el_nr, neigh_el) for neigh_el in t2t[start_el_nr]
            if neigh_el in remaining_els
            ]

    pts = mesh.nodes

    while queue:
        from_el, to_el = queue.pop(-1)

        if to_el not in remaining_els:
            continue

        #       v_from
        #        o._
        #       /   ~._
        #      /       ~._
        #     /           ~._
        # va o---------------o vb
        #     \          _.~
        #      \      _.~
        #       \  _.~
        #        o~
        #       v_to

        from_vertices = set(mesh.elements[from_el])
        to_vertices = set(mesh.elements[to_el])
        shared_edge = from_vertices & to_vertices
        va, vb = shared_edge
        v_to, = to_vertices - shared_edge
        v_from, = from_vertices - shared_edge

        from_normal = np.cross(pts[v_from]-pts[va], pts[vb]-pts[va])
        from_normal /= la.norm(from_normal)

        to_normal = np.cross(pts[vb]-pts[va], pts[v_to]-pts[va])
        to_normal /= la.norm(to_normal)

        from_matches_normal = (
                (la.norm(from_normal - mesh.normals[from_el]) < thresh)
                ^ flip_flags[from_el])

        to_matches_normal = la.norm(to_normal - mesh.normals[to_el]) < thresh

        flip_flags[to_el] = to_matches_normal ^ from_matches_normal
        remaining_els.remove(to_el)

        queue.extend(
                (to_el, neigh_el) for neigh_el in t2t[to_el]
                if neigh_el in remaining_els
                )

def perform_flips(mesh, flip_flags):
    new_elements = np.empty_like(mesh.elements)

    for el in xrange(len(mesh.elements)):
        if flip_flags[el]:
            a,b,c = mesh.elements[el]
            new_elements[el] = a,c,b
        else:
            new_elements[el] = mesh.elements[el]

    return Mesh(mesh.nodes, new_elements)




def generate_face_groups(mesh):
    done = set()

    nb = mesh.triangle_to_neighbor_triangles

    group_starts = set()
    while len(done) < len(mesh.elements):
        group_starts.add(
                (set(xrange(len(mesh.elements)))-done).pop())

        while group_starts:
            group_start = group_starts.pop()
            if group_start in done:
                continue

            face_group = set([group_start])
            done.add(group_start)
            face_normal = mesh.normals[group_start]

            group_candidates = set(nb[group_start])
            seen_for_this_group = set([group_start])
            while group_candidates:
                cand = group_candidates.pop()
                seen_for_this_group.add(cand)
                if cand in done:
                    continue

                if la.norm(mesh.normals[cand] - face_normal) < 1e-10:
                    face_group.add(cand)
                    done.add(cand)
                else:
                    group_starts.add(cand)

                group_candidates.update(nb[cand] - done - seen_for_this_group)

            yield face_normal, face_group





class FaceGridder(object):
    def __init__(self, normal, face_group, mesh, subdiv=100, threshold=1e-10):
        self.normal = normal
        self.face_group = np.fromiter(face_group, dtype=np.intp)

        from hellskitchen.tools import make_basis_from_normal
        temp_basis = make_basis_from_normal(normal)

        temp_basis_vertices = np.dot(temp_basis.T, mesh.nodes.T).T

        face_vertex_indices = set(
                mesh.elements[iel, ivert]
                for iel in face_group
                for ivert in range(3))

        face_vertex_indices_ary = np.fromiter(face_vertex_indices, dtype=np.intp)

        # {{{ corner finding

        def find_a_corner():
            temp_coord_axis = 0

            corner_candidates = None

            while True:
                temp_axis_coords = temp_basis_vertices[:, temp_coord_axis]

                min_temp_coord = np.min(temp_axis_coords[face_vertex_indices_ary])
                corner_candidates_for_axis = np.where(
                        np.abs(temp_axis_coords - min_temp_coord) < 1e-13)[0]

                corner_candidates_for_axis = (
                        set(corner_candidates_for_axis)
                        & face_vertex_indices)

                if corner_candidates is not None:
                    corner_candidates = (
                            corner_candidates & corner_candidates_for_axis)
                else:
                    corner_candidates = corner_candidates_for_axis
                if len(corner_candidates) == 1:
                    return iter(corner_candidates).next()

                temp_coord_axis += 1
                if temp_coord_axis == 2: # that's the normal axis, quit looking
                    raise RuntimeError("corner not found")

        corner_vert_id = find_a_corner()

        # }}}

        # {{{ basis finding

        neigh_use_count = {}

        for corner_tri in mesh.vertex_to_triangle[corner_vert_id]:
            if corner_tri not in face_group:
                continue

            for tri_vert_id in mesh.elements[corner_tri]:
                if tri_vert_id != corner_vert_id:
                    neigh_use_count[tri_vert_id] = neigh_use_count.get(tri_vert_id, 0) + 1

        edge_vertex_id_a, edge_vertex_id_b = [vert_id
                for vert_id, use_count in neigh_use_count.iteritems()
                if use_count == 1]

        vc = mesh.nodes
        self.basis = np.array([
            vc[edge_vertex_id_a] - vc[corner_vert_id],
            vc[edge_vertex_id_b] - vc[corner_vert_id],
            normal]).T
        self.origin = vc[corner_vert_id]

        dual_basis = la.inv(self.basis)

        # }}}

        basis_vertices = (np.dot(dual_basis, mesh.nodes.T).T)[:, :2]

        basis_bbox_min, basis_bbox_max = (
                np.min(basis_vertices[face_vertex_indices_ary], axis=0),
                np.max(basis_vertices[face_vertex_indices_ary], axis=0),
                )
        self.origin_in_basis = basis_bbox_min
        extent = self.extent_in_basis = basis_bbox_max - basis_bbox_min
        self.step = extent/subdiv

        from pytools import ProgressBar

        from pytools.spatial_btree import SpatialBinaryTreeBucket
        tree_root = SpatialBinaryTreeBucket(basis_bbox_min, basis_bbox_max)

        from pytools import Record
        class ElementInfo(Record):
            pass

        el_info = {}
        for iel in face_group:
            el_verts = basis_vertices[mesh.elements[iel]]
            el_bbox_min = np.min(el_verts, axis=0)
            el_bbox_max = np.max(el_verts, axis=0)
            tree_root.insert(iel, (el_bbox_min, el_bbox_max))

            va, vb, vc = el_verts
            bary_basis = np.array([vb-va, vc-va]).T
            bary_dual_basis = la.inv(bary_basis)
            el_info[iel] = ElementInfo(base=va, to_bary=bary_dual_basis,
                    el_bbox_min=el_bbox_min-threshold, el_bbox_max=el_bbox_max+threshold)

        x0, x1 = np.mgrid[0:1:subdiv*1j, 0:1:subdiv*1j]

        from_index = np.zeros(x0.shape, dtype=np.intp)

        low = -threshold
        high = 1+threshold

        def plot_tri(iel):
            import matplotlib.pyplot as pt
            vert_numbers = list(mesh.elements[iel, :])
            vert_numbers.append(vert_numbers[0])
            el_verts = basis_vertices[vert_numbers]
            pt.plot(el_verts[:, 0], el_verts[:, 1], "-")

        pb = ProgressBar("finding face grid", subdiv)
        for i0 in range(subdiv):
            pb.progress()
            for i1 in range(subdiv):
                pt = basis_bbox_min + extent*np.array([x0[i0, i1], x1[i0, i1]])

                found_count = 0
                for iel in tree_root.generate_matches(pt):
                    ei = el_info[iel]

                    if (ei.el_bbox_min <= pt).all() and (pt <= ei.el_bbox_max).all():
                        bc_a, bc_b = np.dot(ei.to_bary, pt-ei.base)
                        bc_a = float(bc_a)
                        bc_b = float(bc_b)
                        bc_c = 1-bc_a-bc_b

                        if (
                                low <= bc_a <= high
                                and
                                low <= bc_b <= high
                                and
                                low <= bc_c <= high):
                            found_count += 1
                            break

                if not found_count:
                    print basis_bbox_min
                    print basis_bbox_max
                    raise RuntimeError("no triangle found for %s" % pt)

                from_index[i0, i1] = iel
        pb.finished()

        self.from_index = from_index

    def __call__(self, field):
        return field[self.from_index]




# vim: foldmethod=marker
