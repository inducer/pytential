"""
Tensor Product Quadrature vs. Vioreanu-Rokhlin Quadrature for Plane Wave on Sphere
==================================================================================

This test compares the absolute error of **Tensor Product Quadrature** and 
**Vioreanu-Rokhlin Quadrature** against the number of discretization nodes
with matched total polynomial degree exactness.

Comparison of Polynomial Exactness
----------------------------------

The following table summarizes the total degree of polynomial exactness of both quadrature methods
based on the order:

Order     VR Exact_to     Tensor Exact_to
-----------------------------------------
1         2              3
2         4              5
3         5              7
4         7              9
5         8              11
6         10             13
7         12             15
8         14             17
9         15             19
10        17             21
11        19             23
12        20             25
13        22             27
14        24             29
15        25             31
16        27             33
17        28             35
18        30             37
19        32             39


Wave Function and Sphere Integral
---------------------------------

The normal direction is:

    d = [-5, 4, 1], n = d / <d, d>

The plane wave is defined as:

    f(x) = exp(1j * n · x)

We compute the integral of the plane wave over a sphere of radius 1:

    ∫_sphere f dS ≈ 10.57423625632583807548

This value is obtained using Mathematica with a working precision of 21 digits.

Mathematica Code
----------------

Below is the Mathematica code used to define the wave function and compute the integral numerically:

    n = Normalize[{-5, 4, 1}]; 
    r = 1; 
    wave[θ_, φ_] := 
      Exp[I r (n[[1]] Sin[θ] Cos[φ] + 
               n[[2]] Sin[θ] Sin[φ] + 
               n[[3]] Cos[θ])];

    NIntegrate[
     wave[θ, φ] * r^2 Sin[θ], 
     {θ, 0, Pi}, 
     {φ, 0, 2 Pi}, 
     WorkingPrecision -> 21
    ]

"""

import meshmode.mesh.generation as mgen
import numpy as np
import matplotlib.pyplot as plt
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureSimplexGroupFactory
from pytential.qbx import QBXLayerPotentialSource
from pytential import GeometryCollection, bind, sym
from arraycontext import flatten
from meshmode.array_context import PyOpenCLArrayContext
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)
actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

def quadrature(level, target_order, qbx_order, tensor=False):
    if tensor:
        from meshmode.mesh import TensorProductElementGroup
        mesh = mgen.generate_sphere(1, target_order, uniform_refinement_rounds=level, group_cls=TensorProductElementGroup)
        from meshmode.discretization.poly_element import InterpolatoryQuadratureGroupFactory
        pre_density_discr = Discretization(actx, mesh, InterpolatoryQuadratureGroupFactory(target_order))
    else:
        mesh = mgen.generate_sphere(1, target_order, uniform_refinement_rounds=level)
        pre_density_discr = Discretization(actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = QBXLayerPotentialSource(pre_density_discr, target_order, qbx_order, fmm_order=False)
    dis_stage = sym.QBX_SOURCE_STAGE1
    places = GeometryCollection({"qbx": qbx}, auto_where=('qbx'))
    density_discr = places.get_discretization("qbx", dis_stage)
    ambient_dim = qbx.ambient_dim
    dofdesc = sym.DOFDescriptor("qbx", dis_stage)

    sources = density_discr.nodes()
    weights_nodes = bind(places, sym.weights_and_area_elements(ambient_dim=3, dim=2, dofdesc=dofdesc))(actx)

    sources_h = actx.to_numpy(flatten(sources, actx)).reshape(ambient_dim, -1)
    weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))

    return sources_h, weights_nodes_h

def wave(x):
    n = np.array([-5, 4, 1])
    n = n / np.linalg.norm(n)
    return np.exp(1j * np.dot(n, x))

def run_test(vr_target_orders, tensor_target_orders, refine_levels):
    ref = 10.57423625632583807548  
    
    for vr_target_order, tensor_target_order in zip(vr_target_orders, tensor_target_orders):
        print(f"{'VR Order'}: {vr_target_order}, Tensor Order: {tensor_target_order}")
        print(f"{'VR Nodes':<15}{'Tensor Nodes':<15}{'VR Error':<20}{'Tensor Error':<20}")
        print("-" * 70)
        vr_result = []
        tensor_result = []
        vr_nodes = []
        tensor_nodes = []
        vr_err = []
        tensor_err = []
        
        for level in refine_levels:
            # VR quadrature
            qbx_order = vr_target_order 
            sources_h, weights_nodes_h = quadrature(level, vr_target_order, qbx_order=qbx_order)
            vr_value = np.dot(wave(sources_h), weights_nodes_h)
            vr_result.append(vr_value)
            vr_nodes.append(len(sources_h[0]))
            vr_err.append(np.abs(vr_value - ref))

            # Tensor quadrature
            qbx_order = tensor_target_order
            sources_h, weights_nodes_h = quadrature(level, tensor_target_order, qbx_order=qbx_order, tensor=True)
            tensor_value = np.dot(wave(sources_h), weights_nodes_h)
            tensor_result.append(tensor_value)
            tensor_nodes.append(len(sources_h[0]))
            tensor_err.append(np.abs(tensor_value - ref))

            print(f"{vr_nodes[-1]:<15}{tensor_nodes[-1]:<15}"
                  f"{vr_err[-1]:<20.12e}{tensor_err[-1]:<20.12e}")
            
            if tensor_err[-1] <= 1e-13 or vr_err[-1] <= 1e-13:
                break
        
        print("\n")
        
        plt.figure()
        plt.semilogy(vr_nodes, vr_err, "o-", label=f"Vioreanu-Rokhlin (Order {vr_target_order})")
        plt.semilogy(tensor_nodes, tensor_err, "o-", label=f"Tensor (Order {tensor_target_order})")
        plt.xlabel(r"$\# \mathrm{nodes}$")
        plt.ylabel(r"$\log_{10}(|\mathrm{abs\ err}|)$")
        plt.legend()
        plt.grid(True)
        plt.title(
            rf"$\log_{{10}}(|\mathrm{{abs\ err}}|) \ \mathrm{{vs}} \ \# \mathrm{{nodes}}$" "\n"
            rf"$\mathrm{{VR\ order}} = {vr_target_order}, \mathrm{{Tensor\ order}} = {tensor_target_order}$"
        )
        plt.show()

if __name__ == "__main__":
    refine_levels = [0, 1, 2, 3, 4, 5, 6]
    vr_target_orders = [4, 9, 16]  
    tensor_target_orders = [3, 7, 13] 
    run_test(vr_target_orders, tensor_target_orders, refine_levels)