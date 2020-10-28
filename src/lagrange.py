#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as df


# get solution, values known at Lagrange nodes, compute at "xi"
def get_sol(p, W, u, xi):
    u_sol = u[0] * basis1d(p, 0, xi)
    for k in range(1, p + 1):
        u_sol += u[k] * basis1d(p, k, xi)

    u_sol_fenics = df.Function(W)
    u_sol_fenics.assign(u_sol)

    return u_sol_fenics


# get solution derivative, values known at Lagrange nodes, compute at "xi"
def get_sol_deriv(p, W, u, xi):
    du_sol = u[0] * basis1d_deriv(p, 0, xi)
    for k in range(1, p + 1):
        du_sol += u[k] * basis1d_deriv(p, k, xi)

    du_sol_fenics = df.Function(W)
    du_sol_fenics.assign(du_sol)

    return du_sol_fenics


# 1d Lagrange basis
def basis1d(p, basis_id, xi):
    if p == 0:
        return zerobasis1d(basis_id, xi)
    elif p == 1:
        return linearbasis1d(basis_id, xi)
    elif p == 2:
        return quadraticbasis1d(basis_id, xi)
    elif p == 3:
        return cubicbasis1d(basis_id, xi)
    elif p == 4:
        return biquadraticbasis1d(basis_id, xi)


# Derivative of 1d Lagrange basis
def basis1d_deriv(p, basis_id, xi):
    if p == 1:
        return linearbasis1d_deriv(basis_id, xi)
    elif p == 2:
        return quadraticbasis1d_deriv(basis_id, xi)
    elif p == 3:
        return cubicbasis1d_deriv(basis_id, xi)


# Constant 1d basis, reference interval (0,1)
# noinspection PyUnusedLocal
def zerobasis1d(basis_id, xi):
    if basis_id == 0:
        return 1


# Linear 1d Lagrange basis, reference interval (0,1)
def linearbasis1d(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return lambda1
    elif basis_id == 1:
        return lambda2


# Derivative of the linear basis
# noinspection PyUnusedLocal
def linearbasis1d_deriv(basis_id, xi):
    if basis_id == 0:
        return -1
    elif basis_id == 1:
        return +1


# Quadratic 1d Lagrange basis, reference interval (0,1)
def quadraticbasis1d(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return lambda1 * (2 * lambda1 - 1)
    elif basis_id == 1:
        return 4 * lambda1 * lambda2
    elif basis_id == 2:
        return lambda2 * (2 * lambda2 - 1)


# Derivative of the quadratic basis
def quadraticbasis1d_deriv(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return -(4 * lambda1 - 1)
    elif basis_id == 1:
        return +4 * (lambda1 - lambda2)
    elif basis_id == 2:
        return +(4 * lambda2 - 1)


# Cubic 1d Lagrange basis, reference interval (0,1)
def cubicbasis1d(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return 1. / 2. * lambda1 * (3 * lambda1 - 2) * (3 * lambda1 - 1)
    elif basis_id == 1:
        return 9. / 2. * lambda1 * lambda2 * (3 * lambda1 - 1)
    elif basis_id == 2:
        return 9. / 2. * lambda1 * lambda2 * (3 * lambda2 - 1)
    elif basis_id == 3:
        return 1. / 2. * lambda2 * (3 * lambda2 - 2) * (3 * lambda2 - 1)


# Derivative of the cubic basis
def cubicbasis1d_deriv(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return -1. / 2. * (27 * lambda1 ** 2 - 18 * lambda1 + 2)
    elif basis_id == 1:
        return +9. / 2. * (3 * lambda1 ** 2 - 6 * lambda1 * lambda2 - lambda1 + lambda2)
    elif basis_id == 2:
        return -9. / 2. * (3 * lambda2 ** 2 - 6 * lambda1 * lambda2 + lambda1 - lambda2)
    elif basis_id == 3:
        return +1. / 2. * (27 * lambda2 ** 2 - 18 * lambda2 + 2)


# Biquadratic 1d Lagrange basis, reference interval (0,1)
def biquadraticbasis1d(basis_id, xi):
    lambda1 = 1 - xi
    lambda2 = xi

    if basis_id == 0:
        return +1. / 3. * lambda1 * (4 * lambda1 - 1) * (4 * lambda2 - 1) * (2 * lambda2 - 1)
    elif basis_id == 1:
        return -16. / 3. * lambda1 * lambda2 * (4 * lambda1 - 1) * (2 * lambda2 - 1)
    elif basis_id == 2:
        return +4 * lambda1 * lambda2 * (4 * lambda1 - 1) * (4 * lambda2 - 1)
    elif basis_id == 3:
        return -16. / 3. * lambda1 * lambda2 * (4 * lambda2 - 1) * (2 * lambda1 - 1)
    elif basis_id == 4:
        return +1. / 3. * lambda2 * (4 * lambda2 - 1) * (4 * lambda1 - 1) * (2 * lambda1 - 1)

# End of file
