#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics
import numpy as np
import mesh_generator.geometry2d as gm
import mesh_generator.bisection_meshes_2d as mgen
import mesh_generator.io_mesh as meshio

from pathlib import Path
cur_dir = str(Path().absolute())
base_mesh_dir = cur_dir + "/meshes/"


def gen_initial_mesh():
    mesh_dir = base_mesh_dir + "square_twoPiecewise/"

    dat_mesh_file = mesh_dir + "mesh_l0.dat"
    vertices, cells = meshio.readMesh2d_dat(dat_mesh_file)

    h5_mesh_file = mesh_dir + "mesh_l0.h5"
    meshio.writeMesh_h5(vertices, cells, h5_mesh_file)


def genNestedMeshes_quasiUniform_unitSquareDomain(bool_mesh_plot, bool_mesh_write):
    mesh_dir = base_mesh_dir + "square_quasiUniform/"

    levels = np.array([1, 2, 3, 4, 5, 6, 7])
    M = 2 ** levels
    print(M)

    for i in range(0, levels.size):

        mesh_file = mesh_dir + "mesh_l" + str(int(levels[i])) + ".h5"
        mesh = fenics.UnitSquareMesh(M[i], M[i])

        print(mesh.hmax(), mesh_file)
        if bool_mesh_plot:
            meshio.plot_mesh(mesh)
        if bool_mesh_write:
            meshio.write_mesh_h5(mesh, mesh_file)


def genNestedMeshes_quasiUniform_gammaShapedDomain(bool_mesh_plot, bool_mesh_write):
    # deg = 1
    h0_init = 0.5
    angle = 1.5 * np.pi

    polygon = gm.AngularDomain(angle)
    mesh_dir = base_mesh_dir + "gammaShaped_quasiUniform/"

    levels = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    h = 1 / (2 ** levels)
    print(h)

    mesh_gen = mgen.BisectionMeshGenerator2d(polygon)
    for i in range(0, levels.size):

        mesh_file = mesh_dir + "mesh_l" + str(levels[i]) + ".h5"
        mesh = mesh_gen.bisection_uniform(h0_init, h[i])

        print(mesh.hmax(), mesh.num_vertices(), '\n', mesh_file)
        if bool_mesh_plot:
            meshio.plot_mesh(mesh)
        if bool_mesh_write:
            meshio.write_mesh_h5(mesh, mesh_file)


def genNestedMeshes_bisecRefined_gammaShapedDomain(bool_mesh_plot, bool_mesh_write):
    deg = 0
    h0_init = 0.5
    angle = 1.5 * np.pi

    zeta = np.pi / angle
    delta = 1 - zeta

    polygon = gm.AngularDomain(angle)

    singular_corners = np.array([2, 3, 4])
    refine_weights = [0, delta, 0]
    # refine_flags = [1]*len(refine_weights)
    refine_flags = [0, 1, 0]

    mesh_dir = None
    if deg == 0:
        mesh_dir = base_mesh_dir + "gammaShaped_bisecRefine/zero/"
    elif deg == 1:
        mesh_dir = base_mesh_dir + "gammaShaped_bisecRefine/linear/"
    elif deg == 2:
        mesh_dir = base_mesh_dir + "gammaShaped_bisecRefine/quadratic/"
    elif deg == 3:
        mesh_dir = base_mesh_dir + "gammaShaped_bisecRefine/cubic/"

    polygon.set_singular_corners(singular_corners)
    polygon.set_refine_weights(refine_weights)
    polygon.set_refine_flags(refine_flags)

    levels = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    h = 1 / (2 ** levels)
    print(deg)
    print(h)

    mesh_gen = mgen.BisectionMeshGenerator2d(polygon)
    # mesh_init = mesh_gen.uniform(h0_init)
    # meshio.plot_mesh(mesh_init)

    for i in range(0, levels.size):

        mesh_file = mesh_dir + "mesh_l" + str(levels[i]) + ".h5"
        mesh = mesh_gen.bisection(deg, h0_init, h[i])

        print(mesh.hmax(), mesh.num_vertices(), '\n', mesh_file)
        if bool_mesh_plot:
            meshio.plot_mesh(mesh)
        if bool_mesh_write:
            meshio.write_mesh_h5(mesh, mesh_file)


def genNestedMeshes_quasiUniform_squareTwoPiecewiseDomains(bool_mesh_plot, bool_mesh_write):
    mesh_dir = base_mesh_dir + "square_twoPiecewise/quasiUniform/"

    levels = np.array([1, 2, 3, 4, 5, 6])
    h = 1 / (2 ** levels)
    print(h)

    init_mesh_file = base_mesh_dir + "square_twoPiecewise/mesh_l0.h5"
    init_mesh = meshio.read_mesh_h5(init_mesh_file)
    if bool_mesh_plot:
        meshio.plot_mesh(init_mesh)

    polygon = []
    bis_refine = mgen.BisectionRefinement(polygon)

    mesh = init_mesh
    print(mesh.hmax(), mesh.hmin(), mesh.num_vertices())
    for i in range(0, levels.size):

        mesh_file = mesh_dir + "mesh_l" + str(levels[i]) + ".h5"
        mesh = bis_refine.uniform(h[i], mesh)

        print(mesh.hmax(), mesh.hmin(), mesh.num_vertices(), '\n', mesh_file)
        if bool_mesh_plot:
            meshio.plot_mesh(mesh)
        if bool_mesh_write:
            meshio.write_mesh_h5(mesh, mesh_file)


def genNestedMeshes_bisecRefined_squareTwoPiecewiseDomains(bool_mesh_plot, bool_mesh_write):
    mesh_dir = base_mesh_dir + "square_twoPiecewise/bisecRefine/"

    deg = 2

    levels = np.array([1, 2, 3, 4, 5])
    h = 1 / (2 ** levels)
    print(h)

    zeta = 0.6  # singular exponent
    delta = 1 - zeta

    singular_corners = np.array([5])
    refine_weights = [delta]
    refine_flags = [1]

    init_mesh_file = base_mesh_dir + "square_twoPiecewise/mesh_l0.h5"
    init_mesh = meshio.read_mesh_h5(init_mesh_file)
    if bool_mesh_plot:
        meshio.plot_mesh(init_mesh)

    if deg == 0:
        mesh_dir = base_mesh_dir + "square_twoPiecewise/bisecRefine/zero/"
    elif deg == 1:
        mesh_dir = base_mesh_dir + "square_twoPiecewise/bisecRefine/linear/"
    elif deg == 2:
        mesh_dir = base_mesh_dir + "square_twoPiecewise/bisecRefine/quadratic/"
    elif deg == 3:
        mesh_dir = base_mesh_dir + "square_twoPiecewise/bisecRefine/cubic/"

    polygon = gm.SquareTwoPiecewiseDomain()

    polygon.set_singular_corners(singular_corners)
    polygon.set_refine_weights(refine_weights)
    polygon.set_refine_flags(refine_flags)

    bis_refine = mgen.BisectionRefinement(polygon)
    mesh = init_mesh
    for i in range(0, levels.size):

        mesh_file = mesh_dir + "mesh_l" + str(levels[i]) + ".h5"
        mesh = bis_refine.uniform(h[i], mesh)
        mesh = bis_refine.local(deg, h[i], mesh)

        print(mesh.hmax(), mesh.num_vertices(), '\n', mesh_file)
        if bool_mesh_plot:
            meshio.plot_mesh(mesh)
        if bool_mesh_write:
            meshio.write_mesh_h5(mesh, mesh_file)


if __name__ == '__main__':
    bool_mesh_plot_ = True
    bool_mesh_write_ = False

    genNestedMeshes_quasiUniform_unitSquareDomain(bool_mesh_plot_, bool_mesh_write_)
    # genNestedMeshes_quasiUniform_gammaShapedDomain(bool_mesh_plot_, bool_mesh_write_)
    # genNestedMeshes_bisecRefined_gammaShapedDomain(bool_mesh_plot_, bool_mesh_write_)

    # gen_initial_mesh()
    # genNestedMeshes_quasiUniform_squareTwoPiecewiseDomains(bool_mesh_plot_, bool_mesh_write_)
    # genNestedMeshes_bisecRefined_squareTwoPiecewiseDomains(bool_mesh_plot_, bool_mesh_write_)

# End of file
