#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as sla
import dolfin as df
from systems.waveO1.local_expressions import ExactSolution2d, ExactSolution1d
from src.time_integrator import TimeIntegrator
import src.legendre as legendre
import src.quad_legendre as quad
import sys


# xt DG projection
class ProjectionXtDG(TimeIntegrator):

    def __init__(self):
        self.tQuad = None
        self.uE = None

        self.test_fun = None
        self.ndof_x = None

    # set system
    def set(self, cfg, test_case, mesh):
        self.set_local(cfg, test_case, mesh)
        self.tQuad = quad.Quad(self.deg_t)

        deg_ref_x = cfg['deg_x_v'] + 2
        if cfg['system'] == "waveO1":
            if self.ndim == 1:
                self.uE = ExactSolution1d(test_case.T, test_case, degree=deg_ref_x)
            elif self.ndim == 2:
                self.uE = ExactSolution2d(test_case.T, test_case, degree=deg_ref_x)
        else:
            sys.exit("\nUnknown system!\n")

    # evaluate projection
    def eval(self, t):

        ne_t = t.shape[0] - 1
        dt = t[1] - t[0]

        # assemble matrices in space
        self.xDiscr.set_system(self.deg_x_v, self.deg_x_sigma, self.test_case, self.mesh)
        mass_x, _ = self.xDiscr.assemble_system()
        sp_mass_x = df.as_backend_type(mass_x).sparray().tocsc()
        lu_Mx = sla.splu(sp_mass_x)

        # assemble matrices in time
        # Mt = np.eye(self.deg_t + 1)

        V = self.xDiscr.FunctionSpace  # function Space
        self.test_fun = df.TestFunction(V)
        self.ndof_x = V.dim()
        ndof = V.dim() * ne_t * (self.deg_t + 1)
        print('       Number of degrees of freedom: ', ndof)

        u = []  # space-time solution
        # proj_mat = sp.sparse.kron(Mt, sp_mass_x).tocsc()

        u_cur = None
        for i in range(0, ne_t):
            rhs = self.assemble_rhs(t[i], t[i + 1])
            # u_cur = sla.spsolve(proj_mat, rhs)
            # u_old = u_cur.copy()

            u_cur = np.zeros(self.ndof_x * (self.deg_t + 1))
            for j in range(0, self.deg_t + 1):
                u_cur[j * self.ndof_x: (j + 1) * self.ndof_x] = lu_Mx.solve(
                    rhs[j * self.ndof_x: (j + 1) * self.ndof_x])
            # u_old = u_cur.copy()

            # save solution at t=t_n
            if self.save_xt_sol:
                self.write_xtSol(u, u_cur, dt)

        if self.save_xt_sol:
            return u, ndof

        uSol_tn = df.Function(V)
        uSol_tn.vector().set_local(self.gen_sol(u_cur, dt, 1))
        return uSol_tn, ndof

    # compute xt-DG projection
    def assemble_rhs(self, t_nm1, t_n):
        u = self.tQuad.projectionL2(self.ndof_x, self.projectionL2_xDG, t_nm1, t_n)
        return u

    # spatial projection
    def projectionL2_xDG(self, t):
        self.uE.t = t
        return df.assemble(df.inner(self.uE, self.test_fun) * df.dx)

    # generate solution at t=t_{n-1} + xi*h for xi \in [0,1], given u(t) for t \in [t_{n-1}, t_{n}]
    def gen_sol(self, u, h, xi):

        u_ = np.zeros(self.ndof_x)
        for k in range(0, self.deg_t + 1):
            basisVal = legendre.basis1d(k, xi) / np.sqrt(h)
            u_ += u[k * self.ndof_x: (k + 1) * self.ndof_x] * basisVal

        return u_

    # write space-time solution at the Lagrange nodes
    def write_xtSol(self, u, u_old, dt):

        for k in range(0, self.deg_t + 1):
            xi = (1. / self.deg_t) * k
            uSol = df.Function(self.xDiscr.FunctionSpace)
            uSol.vector().set_local(self.gen_sol(u_old, dt, xi))
            u.append(uSol)

# End of file
