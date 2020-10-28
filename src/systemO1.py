#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import src.error as err
import src.main_2d as main2d
from src.time_integrator import get_time_integrator
import test_cases.common as test_cases
import utilities as utl


# get mesh and solution directories
def get_directories(cfg, test_case):
    type_bi = ["zero", "linear", "quadratic", "cubic"]

    dir_mesh = None
    dir_sol = None

    if cfg['mesh type'] == "":
        dir_sol = test_case.dir_sol_quasiUniform

    elif cfg['mesh type'] == "quasiUniform":
        dir_mesh = test_case.dir_mesh_quasiUniform
        dir_sol = test_case.dir_sol_quasiUniform

    elif cfg['mesh type'] == "bisectionRefined":
        dir_mesh = test_case.dir_mesh_bisecRefined + type_bi[cfg['deg_x_sigma']] + "/"
        dir_sol = test_case.dir_sol_bisecRefined

    return dir_mesh, dir_sol


# set filename
def get_filename(cfg):
    if cfg['save xt sol']:
        filename = cfg['output dir'] \
                   + cfg['output filename'] + '_' + cfg['mesh type'] + '_' \
                   + "xtErr" + cfg['error type'] + '_' \
                   + 'degx' + str(cfg['deg_x_v']) + '_degt' + str(cfg['deg_t']) + '.txt'
    else:
        filename = cfg['output dir'] \
                   + cfg['output filename'] + '_' + cfg['mesh type'] + '_' \
                   + "xErr" + cfg['error type'] + '_' \
                   + 'degx' + str(cfg['deg_x_v']) + '_degt' + str(cfg['deg_t']) + '.txt'

    return filename


#########
#   2D  #
#########

# Run solver full-grid
def runFG_2d(cfg, solver, lx, lt):
    ne_tMesh = 2 ** lt

    test_case = test_cases.get(cfg)
    dir_mesh, dir_sol = get_directories(cfg, test_case)
    print("\nMesh directory: ", dir_mesh)
    print("Solution directory: ", dir_sol)

    print('\nRunning full-grid...')
    integrator = get_time_integrator(cfg['time integrator'])
    tMesh, xMesh, u, ndof = solver(cfg, dir_mesh, test_case, integrator, lx, ne_tMesh)
    print('Meshwidth: ', xMesh.hmax())

    return u, tMesh, xMesh


# Run solver sparse-grid 2d
def runSG_2d(cfg, solverFG, L, L0):
    test_case = test_cases.get(cfg)
    dir_mesh, dir_sol = get_directories(cfg, test_case)
    print("\nMesh directory: ", dir_mesh)
    print("Solution directory: ", dir_sol)

    print('\nRunning sparse-grid...')
    integrator = get_time_integrator(cfg['time integrator'])
    u_solutions, tMeshes, xMeshes, sparse_grids, ndof = main2d.SG_2d(cfg, dir_mesh, test_case, solverFG, integrator,
                                                                     L0, L, L0, L)

    print('\n  Compute sparse-grid solution error...\n')
    sparse_grids.set(xMeshes, tMeshes)
    errSol_norm = sparse_grids.eval_error(u_solutions)

    print('Number of degrees of freedom: ', ndof)
    print('Error: ', errSol_norm)

    return u_solutions, tMeshes, xMeshes


# Run convergence test for solver full-grid 2d
def convergenceFG_2d(cfg, solver, lx, lt):
    test_case = test_cases.get(cfg)
    dir_mesh, dir_sol = get_directories(cfg, test_case)
    print("\nMesh directory: ", dir_mesh)
    print("Solution directory: ", dir_sol)

    ne_xMesh = 2 ** lx
    ne_tMesh = 2 ** lt

    integrator = get_time_integrator(cfg['time integrator'])
    error = err.Error(cfg, dir_mesh, test_case)

    print('\nRunning convergence test full-grid...')
    ndof = [0] * (lx.shape[0])
    errSol_norm = np.zeros((lx.shape[0], 2))
    for k in range(0, lx.shape[0]):
        tMesh, xMesh, u, ndof[k] = solver(cfg, dir_mesh, test_case, integrator, lx[k], ne_tMesh[k])

        error.set_system(xMesh)
        errSol_norm[k, :] = error.eval(tMesh, u)
        print('  Number of degrees of freedom: ', ndof[k])
        print('  Error: ', errSol_norm[k])

    print('\nNumber of degrees of freedom: ', ndof)
    print('inv(hx): ', ne_xMesh)
    print('inv(ht): ', ne_tMesh)
    print('Error:\n', errSol_norm)

    if cfg['write output']:
        filename = get_filename(cfg)
        print('\n\nWrite output to file: ', filename, '\n\n')
        utl.write_outputFG(filename, cfg, ne_xMesh, ne_tMesh, ndof, errSol_norm)

    return errSol_norm, np.asarray(ndof)


# Run convergence test for solver sparse-grid 2d
def convergenceSG_2d(cfg, solverFG, Lx, L0x, Lt, L0t):
    test_case = test_cases.get(cfg)
    dir_mesh, dir_sol = get_directories(cfg, test_case)
    print("\nMesh directory: ", dir_mesh)
    print("Solution directory: ", dir_sol)

    invH = 2 ** (Lx + 1)

    integrator = get_time_integrator(cfg['time integrator'])

    print('\nRunning convergence test sparse-grid...')
    ndof = [0] * (Lx.shape[0])
    errSol_norm = np.zeros((Lx.shape[0], 2))
    for k in range(0, Lx.shape[0]):
        u_solutions, tMeshes, xMeshes, sparse_grids, ndof[k] = main2d.SG_2d(cfg, dir_mesh, test_case, solverFG,
                                                                            integrator, L0x, Lx[k], L0t, Lt[k])

        sparse_grids.set(xMeshes, tMeshes)
        errSol_norm[k, :] = sparse_grids.eval_error(u_solutions)
        print('  Number of degrees of freedom: ', ndof[k])
        print('  Error: ', errSol_norm[k])

    print('\nNumber of degrees of freedom: ', ndof)
    print('inv(h): ', invH)
    print('Error:\n', errSol_norm)

    if cfg['write output']:
        filename = get_filename(cfg)
        print('\n\nWrite output to file: ', filename, '\n\n')
        utl.write_outputSG(filename, cfg, invH, ndof, errSol_norm)

    return errSol_norm, np.asarray(ndof)

# End of file
