#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import src.functions as fns
import src.sparse_grids as spg
from src.xtDG_projection import ProjectionXtDG
import mesh_generator.io_mesh as meshio


# projection of exact solution, full-grid 2d
# noinspection PyUnusedLocal
def projectionFG_2d(cfg, dir_mesh, test_case, integrator, lx, ne_tMesh):
    xMesh_file = dir_mesh + 'mesh_l%d.h5' % lx
    print('\n    Read mesh file: ', xMesh_file)
    print('    Uniform time series, number of intervals: ', ne_tMesh)

    xMesh = meshio.read_mesh_h5(xMesh_file)
    tMesh = fns.get_uniform_time_series(test_case.T, ne_tMesh)

    xtDGproj = ProjectionXtDG()
    xtDGproj.set(cfg, test_case, xMesh)
    u, ndof = xtDGproj.eval(tMesh)

    return tMesh, xMesh, u, ndof


# time integrator + spatial_discretisation solver, full-grid 2d
def schemeFG_2d(cfg, dir_mesh, test_case, integrator, lx, ne_tMesh):
    xMesh_file = dir_mesh + 'mesh_l%d.h5' % lx
    print('\n    Read mesh file: ', xMesh_file)
    print('    Uniform time series, number of intervals: ', ne_tMesh)

    xMesh = meshio.read_mesh_h5(xMesh_file)
    tMesh = fns.get_uniform_time_series(test_case.T, ne_tMesh)

    cfg.update({'mesh directory': dir_mesh, 'mesh level': lx})
    integrator.set(cfg, test_case, xMesh)
    u, ndof = integrator.run(tMesh)

    return tMesh, xMesh, u, ndof


# time integrator + xdG solver, sparse-grid 2d
def SG_2d(cfg, dir_mesh, test_case, solverFG, integrator, L0x, Lx, L0t, Lt):
    sparse_grids = spg.SparseGrids()
    sparse_grids.set_init(cfg, dir_mesh, test_case, L0x, Lx, L0t, Lt)
    print('\n  Sparse space-time levels:\n', sparse_grids.levels)

    nSG = sparse_grids.levels.shape[0]

    u_solutions = []
    tMeshes = []
    xMeshes = []

    ndof = np.zeros(nSG, dtype=np.int32)
    for k in range(0, nSG):

        lx = sparse_grids.levels[k, 0] + 1
        lt = sparse_grids.levels[k, 1] + 1

        ne_tMesh = 2 ** lt
        tMesh, xMesh, u, ndof[k] = solverFG(cfg, dir_mesh, test_case, integrator, lx, ne_tMesh)

        u_solutions.append(u)
        if k <= Lx - L0x:
            xMeshes.append(xMesh)
            tMeshes.append(tMesh)

    return u_solutions, tMeshes, xMeshes, sparse_grids, np.sum(ndof)

# End of file
