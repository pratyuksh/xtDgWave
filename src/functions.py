#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import src.legendre as legendre


# uniform time series: h_t = T/ne_t
def get_uniform_time_series(T, ne_t):
    tMesh = np.zeros(ne_t + 1)
    for k in range(0, ne_t + 1):
        tMesh[k] = k * T / ne_t

    return tMesh


# Affine mapping, reference interval (0,1) to _specified_ interval
def affineMap(a, b, xi):
    z = a * (1 - xi) + b * xi
    return z


# Build solution
def getSol(p, u, xi):
    val = 0
    for i in range(0, p + 1):
        val += u[i] * legendre.basis1d(i, xi)
    return val


# Build solution gradient
def getSolGrad(p, u, xi):
    val = 0
    for i in range(0, p + 1):
        val += u[i] * legendre.basis1d_deriv(i, xi)
    return val

# End of file
