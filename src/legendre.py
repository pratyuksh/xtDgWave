#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df


# 1d ortho-normalized Gauss-Legendre basis, reference interval (0,1)
def basis1d(pid, xi):
    z = 2 * xi - 1
    if pid == 0:
        return 1
    elif pid == 1:
        return np.sqrt(3) * z
    elif pid == 2:
        return np.sqrt(5) * 0.5 * (3 * z ** 2 - 1)
    elif pid == 3:
        return np.sqrt(7) * 0.5 * (5 * z ** 3 - 3 * z)
    elif pid == 4:
        return np.sqrt(9) * 0.125 * (35 * z ** 4 - 30 * z ** 2 + 3)


# Derivative of 1d ortho-normalized Gauss-Legendre basis, reference interval (0,1)
def basis1d_deriv(pid, xi):
    z = 2 * xi - 1
    if pid == 0:
        return 0
    elif pid == 1:
        return 2 * np.sqrt(3)
    elif pid == 2:
        return 6 * np.sqrt(5) * z
    elif pid == 3:
        return np.sqrt(7) * (15 * z ** 2 - 3)
    elif pid == 4:
        return np.sqrt(9) * (35 * z ** 3 - 15 * z)


# get solution, Legendre coefficients given, compute at "xi"
def get_sol(p, W, u, xi):
    u_sol = u[0] * basis1d(0, xi)
    for k in range(1, p + 1):
        u_sol += u[k] * basis1d(k, xi)

    u_sol_fenics = df.Function(W)
    u_sol_fenics.assign(u_sol)

    return u_sol_fenics

# End of file
