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
tIntegrators = ["dG", "Crank-Nicolson"]
meshes = ["quasiUniform", "bisectionRefined"]


def call_runFG(stabParamsType):
    deg_x = 4
    deg_t = 1
    test_case_name = "test1_smooth_squareDomain"

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
           'save xt sol': False,
           'write output': False,
           'output dir': "output/waveO1/",
           'output filename': "runFG_test3",
           'error type': "L2L2",
           'bool measure signal': True,
           'bool write signal': True,
           'signal outFile': "output/scattering_signal.txt",
           'dump sol': True,
           'dump sol subdir': "test3/",
           'dump sol at time': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
           }

    util.print_config(cfg)

    lx = 4
    lt = 6
    system.runFG_2d(cfg, main.schemeFG_2d, lx, lt)


if __name__ == "__main__":
    stabParamsType_ = 4
    call_runFG(stabParamsType_)

# End of file
