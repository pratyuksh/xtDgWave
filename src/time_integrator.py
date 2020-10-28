#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sciSp
import scipy.sparse.linalg as sla
import dolfin as df
import sys

import src.legendre as legendre
import src.quad_legendre as quad_legendre
import src.quad_lagrange as quad_lagrange
import systems.spatial_discretisation as spatial_discretisation
import io_sol as solio


def get_time_integrator(integrator_type):
    if integrator_type == "Crank-Nicolson":
        return CrankNicolson()

    elif integrator_type == "dG":
        return DGTimeStepping()

    else:
        sys.exit("\nTime integrator: "+integrator_type+" not available!\n")


# Base class: Time integrator
class TimeIntegrator:

    def __init__(self):

        self.cfg = None

        self.deg_x = None
        self.deg_x_v = None
        self.deg_x_sigma = None

        self.deg_t = None
        self.test_case = None
        self.mesh = None

        self.save_xt_sol = None

        self.ndim = None
        self.xDiscr = None

        self.bool_measure_signal = None
        self.bool_write_signal = None
        self.signal_outputFile = None

        self.bool_dump_sol = None
        self.dump_sol_at_time = None
        self.sol_files = None

        self.pardisoSolver = None
        self.dxSub = None

    def set_local(self, cfg, test_case, mesh):

        self.cfg = cfg

        if cfg['system'] == "waveO1":
            self.deg_x_v = cfg['deg_x_v']
            self.deg_x_sigma = cfg['deg_x_sigma']

        elif cfg['system'] == "waveO2":
            self.deg_x = cfg['deg_x']

        self.deg_t = cfg['deg_t']
        self.test_case = test_case
        self.mesh = mesh
        self.save_xt_sol = cfg['save xt sol']

        self.ndim = test_case.ndim
        self.xDiscr = spatial_discretisation.get(cfg)

        # settings for measuring solution signal
        try:
            self.bool_measure_signal = cfg["bool measure signal"]
        except KeyError:
            self.bool_measure_signal = False

        if self.bool_measure_signal:
            try:
                self.bool_write_signal = cfg["bool write signal"]
            except KeyError:
                self.bool_write_signal = False

            if self.bool_write_signal:
                try:
                    self.signal_outputFile = cfg["signal outFile"]
                except KeyError:
                    self.signal_outputFile = "output/signal_out.txt"

        else:
            self.bool_write_signal = False

        # settings for solution output
        try:
            self.bool_dump_sol = cfg["dump sol"]
        except KeyError:
            self.bool_dump_sol = False

        if self.bool_dump_sol:
            try:
                self.dump_sol_at_time = cfg["dump sol at time"]
            except KeyError:
                self.dump_sol_at_time = np.array([test_case.T])

            self.sol_files = []
            for s in range(0, self.dump_sol_at_time.size):
                tOut = self.dump_sol_at_time[s]
                sol_file = 'output/square_twoPiecewise/' + cfg['dump sol subdir'] + cfg[
                    'test case'] + "_t%4.2E.pvd" % tOut
                self.sol_files.append(sol_file)

    # function interpolation to given FunctionSpace
    # noinspection PyMethodMayBeStatic
    def eval_fun(self, fun, V):
        return df.interpolate(fun, V)

    # assemble (f*v), given a FunctionSpace
    # noinspection PyMethodMayBeStatic
    def assemble_fun(self, fun, V):
        v = df.TestFunction(V)
        return df.assemble(df.inner(fun, v) * df.dx)

    def solve_pardiso(self, b):
        return self.pardisoSolver.solve(b)

    def final_pardiso(self):
        self.pardisoSolver.finalize()

    # measuring solution in marked sub-domain
    def mark_subdomains(self):

        # mark cells containing this point
        z = df.Point(1, 0.25)

        subdomain_markers = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        subdomain_markers.set_all(0)

        for cell in df.cells(self.mesh):
            if cell.contains(z):
                subdomain_markers[cell] = 1

        self.dxSub = df.Measure('dx', domain=self.mesh, subdomain_data=subdomain_markers)

    # write signal time series to file
    def write_signal(self, signal):

        Np = signal.shape[0]
        file = open(self.signal_outputFile, "w")
        file.write("%d\n" % Np)
        for k in range(0, Np):  # time data
            if k == Np - 1:
                file.write("%E\n" % signal[k, 0])
            else:
                file.write("%E, " % signal[k, 0])
        for k in range(0, Np):  # signal data
            if k == Np - 1:
                file.write("%E\n" % signal[k, 1])
            else:
                file.write("%E, " % signal[k, 1])
        file.close()


# Derived class: Crank-Nicolson
# Base class Time integrator
class CrankNicolson(TimeIntegrator):

    # set integrator
    def set(self, cfg, test_case, mesh):
        self.set_local(cfg, test_case, mesh)

    # run integrator
    def run(self, t):

        nn_t = t.shape[0]
        dt = t[1] - t[0]

        self.xDiscr.set_system(self.cfg, self.test_case, self.mesh, dt)
        mass, stiff = self.xDiscr.assemble_system()

        V = self.xDiscr.FunctionSpace
        ndof = V.dim() * (nn_t - 1)
        print('       Number of degrees of freedom: ', ndof)

        u = []  # space-time solution
        u_old = df.Function(V)  # solution variable at t_n
        u_old.assign(self.eval_fun(self.xDiscr.init_sol, V))  # initial solution

        # save solution at t=0
        if self.save_xt_sol:
            u.append(u_old.copy())

        sys_mat = mass + 0.5 * dt * stiff
        self.xDiscr.apply_dirichletBCs(sys_mat, t[0])
        xRhs_old = self.xDiscr.assemble_rhs(t[0])

        u_cur = None
        lu_sys_mat = df.LUSolver(sys_mat)
        lu_sys_mat.parameters['reuse_factorization'] = True
        for k in range(1, nn_t):

            xRhs_cur = self.xDiscr.assemble_rhs(t[k])
            rhs = (mass - 0.5 * dt * stiff) * u_old.vector() + 0.5 * dt * (xRhs_cur + xRhs_old)

            u_cur = df.Function(V)
            self.xDiscr.apply_dirichletBCs(rhs, t[1])
            lu_sys_mat.solve(u_cur.vector(), rhs)

            # save solution at t[k]
            if self.save_xt_sol:
                u.append(u_cur.copy())

            xRhs_old = xRhs_cur.copy()
            u_old.assign(u_cur)

        if self.save_xt_sol:
            return u, ndof

        return u_cur, ndof


# Derived class: DG time stepping
# Base class Time integrator
class DGTimeStepping(TimeIntegrator):

    def __init__(self):
        self.tQuad = None

        self.ndof_x = None
        self.rhs_tMat = None

    # set integrator
    def set(self, cfg, test_case, mesh):
        self.set_local(cfg, test_case, mesh)
        self.tQuad = quad_legendre.Quad(self.deg_t)

    # set linear solver
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def set_linear_solver(self, Mt, Et, Ct_plus, Mx, Kx):

        print(Et + Ct_plus, "\n\n")

        eigVals, eigVecs = la.eig(Et + Ct_plus)
        print(eigVals)
        print(eigVecs, "\n\n")

        U, Q = sp.linalg.schur(Et + Ct_plus, 'complex')
        print(U)
        print(Q)

    # run integrator
    def run(self, t):

        # df.parameters.reorder_dofs_serial = False

        ne_t = t.shape[0] - 1
        dt = t[1] - t[0]

        # assemble matrices in space
        self.xDiscr.set_system(self.cfg, self.test_case, self.mesh, dt)
        mass_x, convec_x = self.xDiscr.assemble_system()
        sp_mass_x = df.as_backend_type(mass_x).sparray()
        del mass_x
        sp_convec_x = df.as_backend_type(convec_x).sparray()
        del convec_x

        # assemble matrices in time
        Mt = np.eye(self.deg_t + 1)
        Et = (1. / dt) * self.assemble_evolution_refEl()
        Ct_plus = self.assemble_temporal_numFlux_plus(dt)
        Ct_minus = self.assemble_temporal_numFlux_minus(dt)

        # self.set_linear_solver(Mt, Et, Ct_plus, sp_mass_x, sp_convec_x)

        V = self.xDiscr.FunctionSpace  # function Space
        self.ndof_x = V.dim()
        ndof = V.dim() * ne_t * (self.deg_t + 1)
        print('       Number of degrees of freedom: ', ndof)

        u = []  # space-time solution
        sys_mat = sp.sparse.kron(Et + Ct_plus, sp_mass_x).tocsc() + sp.sparse.kron(Mt, sp_convec_x).tocsc()
        del sp_convec_x
        self.rhs_tMat = sp.sparse.kron(Ct_minus, sp_mass_x)
        del sp_mass_x
        self.xDiscr.apply_dirichletBCs(sys_mat, t[0])
        print("Size of system: ", sys_mat.shape)

        # compute solution in temporal element [t_0, t_1]
        u0 = self.assemble_fun(self.xDiscr.init_sol, V)  # initial solution
        rhs_init = self.assemble_rhs_init(t[0], t[1], u0.get_local())
        self.xDiscr.apply_dirichletBCs(rhs_init, t[1])

        lu_sys_mat = sla.splu(sys_mat)
        u_old = lu_sys_mat.solve(rhs_init)

        signal = None
        if self.bool_measure_signal:
            self.mark_subdomains()
            signal = np.zeros((ne_t * 10, 2))
            signal[0:10, :] = self.measure_signal(u_old, t[0], dt)

        dump_sol_count = None
        if self.bool_dump_sol:
            dump_sol_count = 0
            dump_sol_count = self.dump_solOut(u_old, t[0], dt, dump_sol_count)

        # save solution at t=0,dt
        if self.save_xt_sol:
            self.write_xtSol(u, u_old, dt)

        u_cur = None
        for k in range(1, ne_t):
            rhs = self.assemble_rhs(t[k], t[k + 1], u_old)
            self.xDiscr.apply_dirichletBCs(rhs, t[k])

            u_cur = lu_sys_mat.solve(rhs)
            u_old = u_cur.copy()

            # measure signal in specified region
            if self.bool_measure_signal:
                signal[k * 10:(k + 1) * 10, :] = self.measure_signal(u_cur, t[k], dt)

            # dump solution to file
            if self.bool_dump_sol:
                dump_sol_count = self.dump_solOut(u_cur, t[k], dt, dump_sol_count)

            # save solution at t=t_n
            if self.save_xt_sol:
                self.write_xtSol(u, u_cur, dt)

        if self.save_xt_sol:
            return u, ndof

        if self.bool_write_signal:
            self.write_signal(signal)

        uSol_tn = df.Function(V)
        uSol_tn.vector().set_local(self.gen_sol(u_cur, dt, 1))
        return uSol_tn, ndof

    # Assemble: _temporal_ evolution form on the reference interval [0,1]
    # Et = Et_refEl/ h; h is the interval size
    def assemble_evolution_refEl(self):

        Et_refEl = np.zeros((self.deg_t + 1, self.deg_t + 1))
        for j in range(0, self.deg_t + 1):
            for i in range(0, self.deg_t + 1):
                Et_refEl[j, i] = self.tQuad.evolution_refEl(j, i)

        return Et_refEl

    # Assembly: Temporal flux matrix
    # Assemble: _temporal_ numerical flux form (u_plus v_plus) on the interface at t=tn
    def assemble_temporal_numFlux_plus(self, h):

        Ct_plus = np.zeros((self.deg_t + 1, self.deg_t + 1))
        for j in range(0, self.deg_t + 1):
            for i in range(0, self.deg_t + 1):
                Ct_plus[j, i] = legendre.basis1d(j, 0) * legendre.basis1d(i, 0) / h

        return Ct_plus

    # Assemble: _temporal_ numerical flux form (u_minus v_plus) on the interface at t=tn
    def assemble_temporal_numFlux_minus(self, h):

        Ct_minus = np.zeros((self.deg_t + 1, self.deg_t + 1))
        for j in range(0, self.deg_t + 1):
            for i in range(0, self.deg_t + 1):
                Ct_minus[j, i] = legendre.basis1d(j, 0) * legendre.basis1d(i, 1) / h

        return Ct_minus

    # Assemble: l(v) = source + u0*v_{'+'} @ t=0
    def assemble_rhs_init(self, t0, t1, u0):

        h = t1 - t0
        b1 = self.tQuad.projectionL2(self.ndof_x, self.xDiscr.assemble_rhs, t0, t1)
        b2 = np.zeros(self.ndof_x * (self.deg_t + 1))
        for k in range(0, self.deg_t + 1):
            basisVal = legendre.basis1d(k, 0) / np.sqrt(h)
            b2[k * self.ndof_x: (k + 1) * self.ndof_x] = u0 * basisVal

        return b1 + b2

    # Assemble: l(v) = source + C_minus*u_{n-1}
    def assemble_rhs(self, t_nm1, t_n, u_nm1):
        b = self.tQuad.projectionL2(self.ndof_x, self.xDiscr.assemble_rhs, t_nm1, t_n)
        return b + self.rhs_tMat.dot(u_nm1)

    # generate solution at t=t_{n-1} + xi*h for xi \in [0,1], given u(t) for t \in [t_{n-1}, t_{n}]
    def gen_sol(self, u, h, xi):

        u_ = np.zeros(self.ndof_x)
        for k in range(0, self.deg_t + 1):
            basisVal = legendre.basis1d(k, xi) / np.sqrt(h)
            u_ += u[k * self.ndof_x: (k + 1) * self.ndof_x] * basisVal

        return u_

    # write space-time solution at the Lagrange nodes
    def write_xtSol(self, u, u_old, dt):

        V = self.xDiscr.FunctionSpace  # function Space
        for k in range(0, self.deg_t + 1):
            xi = (1. / self.deg_t) * k
            uSol = df.Function(V)
            uSol.vector().set_local(self.gen_sol(u_old, dt, xi))
            u.append(uSol)

    def measure_signal(self, u, tn, dt):

        xi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        val = np.zeros((xi.size, 2))

        V = self.xDiscr.FunctionSpace  # function Space
        uSol = df.Function(V)

        for k in range(0, xi.size):
            uSol.vector().set_local(self.gen_sol(u, dt, xi[k]))
            val[k, 0] = tn + dt * xi[k]
            val[k, 1] = df.assemble((uSol[0]) * self.dxSub(1))
            print(val[k, 0], val[k, 1])

        return val

    def dump_solOut(self, u, tn, dt, dump_sol_count):

        V = self.xDiscr.FunctionSpace  # function Space
        V_fine = self.xDiscr.FunctionSpace_fine
        dump_sol_maxCount = self.dump_sol_at_time.size

        tol = 1E-8
        for s in range(dump_sol_count, dump_sol_maxCount):
            tOut = self.dump_sol_at_time[s]
            xi = (tOut - tn) / dt
            # print(s, tOut, tn, dt, xi)
            if 0 < xi <= 1 + tol:
                uOut = df.Function(V)
                uOut.vector().set_local(self.gen_sol(u, dt, xi))
                # solio.plot_sol2d(uOut[0])
                uOut_fine = df.interpolate(uOut, V_fine)
                # solio.plot_sol2d(uOut_fine[0])
                sol_file = self.sol_files[dump_sol_count]
                print('Local node value: ', xi)
                print(' Write solution to file: ', sol_file)
                solio.write_sol_pvd(uOut_fine, tOut, sol_file)
                dump_sol_count += 1

        return dump_sol_count

# End of file
