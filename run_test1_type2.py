#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import src.main_2d as main
import src.systemO1 as system
import utilities as util

# System
ndim = 2
system_type = "waveO1"
xDiscretisation_type = "dG"
tIntegrators = ["dG"]
meshes = ["quasiUniform", "bisectionRefined"]
errorTypes = ["L2L2", "DG"]


# Full-grid run; smooth solution
def call_runFG_smooth(deg_x, deg_t, stabParamsType, errorType, saveXtSol):
    test_case_name = "test1_smooth_unitSquareDomain"

    cfg = {'ndim': ndim,
           'system': system_type,
           'test case': test_case_name,
           'deg_x_v': deg_x,
           'deg_x_sigma': deg_x - 1,
           'deg_t': deg_t,
           'time integrator': tIntegrators[0],
           'spatial discretisation': xDiscretisation_type,
           'mesh type': meshes[0],
           'stab params type': stabParamsType,
           'ref xMesh level': -1,
           'error type': errorType,
           'save xt sol': saveXtSol,
           'write output': True,
           'output dir': "output/waveO1/type2/stabParams" + str(stabParamsType) + "/",
           'output filename': "runFG_test1",
           }

    util.print_config(cfg)

    levels = np.array([1, 2, 3, 4, 5, 6])
    lx = levels
    lt = levels

    errL2, ndof = system.convergenceFG_2d(cfg, main.schemeFG_2d, lx, lt)

    return errL2, ndof


# Sparse-grid run; smooth solution
def call_runSG_smooth(deg_x, deg_t, stabParamsType, errorType, saveXtSol):
    test_case_name = "test1_smooth_unitSquareDomain"

    cfg = {'ndim': ndim,
           'system': system_type,
           'test case': test_case_name,
           'deg_x_v': deg_x,
           'deg_x_sigma': deg_x - 1,
           'deg_t': deg_t,
           'time integrator': tIntegrators[0],
           'spatial discretisation': xDiscretisation_type,
           'mesh type': meshes[0],
           'stab params type': stabParamsType,
           'ref xMesh level': -1,
           'error type': errorType,
           'save xt sol': saveXtSol,
           'write output': True,
           'output dir': "output/waveO1/type2/stabParams" + str(stabParamsType) + "/",
           'output filename': "runSG_test1"
           }

    util.print_config(cfg)

    L0x = 0
    Lx = np.array([1, 2, 3, 4, 5])

    L0t = 1
    Lt = np.array([2, 3, 4, 5, 6])

    errL2, ndof = system.convergenceSG_2d(cfg, main.schemeFG_2d, Lx, L0x, Lt, L0t)

    return errL2, ndof


def run_experimentsFG_smooth(errorType, saveXtSol):
    stabParamsTypesFG = [1, 2, 3, 4]
    degsFG = [1, 2, 3]

    for deg in degsFG:
        for stabParamsType in stabParamsTypesFG:
            call_runFG_smooth(deg, deg, stabParamsType, errorType, saveXtSol)


def run_experimentsSG_smooth(errorType, saveXtSol):
    stabParamsTypesSG = [1, 2, 3, 4, 5]
    degsSG = [1, 2]

    for deg in degsSG:
        for stabParamsType in stabParamsTypesSG:
            call_runSG_smooth(deg, deg, stabParamsType, errorType, saveXtSol)


if __name__ == "__main__":

    err_FG, ndof_FG = call_runFG_smooth(1, 1, 1, errorTypes[0], False)
    # err_SG, ndof_SG = call_runSG_smooth(1, 1, 1, errorTypes[0], False)

    errorType_ = errorTypes[0]
    saveXtSol_ = False
    # run_experimentsFG_smooth(errorType_, saveXtSol_)
    # run_experimentsSG_smooth(errorType_, saveXtSol_)

# End of file
