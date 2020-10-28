#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df
import sys

import src.functions as fns
import src.lagrange as lagrange
import systems.waveO1.local_expressions as waveO1_locexp
import mesh_generator.io_mesh as meshio


# Sparse solver
class SparseGrids:

    def __init__(self):

        self.nodes1d = None
        self.weights1d = None

        self.ndim = None
        self.system_name = None
        self.test_case = None
        self.save_xt_sol = None
        self.integrator_type = None
        self.errType = None
        self.deg_t = None

        self.dir_mesh = None

        self.deg_x_v = None
        self.deg_x_sigma = None
        self.deg_ref_x_v = None
        self.deg_ref_x_sigma = None
        self.deg_x = None
        self.deg_ref_x = None

        self.ref_lx = None
        self.L = None
        self.L0 = None
        self.levels = None
        self.comb_coeffs = None

        self.nne_t = None
        self.get_time_index_shift = None
        self.compute = None

        self.tMeshes = None
        self.spaces = None
        self.xMesh_finest = None
        self.W_ref = None
        self.uE = None
        self.duEdt = None

    # Set quadrature nodes and weights
    def set_quad_nodes_weights(self):

        # Gauss-Legendre nodes and weights, for temporal quadrature
        if self.deg_t == 1:
            self.nodes1d = np.array([(1 - 1 / np.sqrt(3)) / 2., (1 + 1 / np.sqrt(3)) / 2.])  # Order 4
            self.weights1d = np.array([1. / 2., 1. / 2.])

        elif self.deg_t == 2:
            self.nodes1d = np.array([(1 - np.sqrt(3. / 5.)) / 2., 0.5, (1 + np.sqrt(3. / 5.)) / 2.])  # Order 6
            self.weights1d = np.array([5. / 18., 4. / 9., 5. / 18.])

        elif self.deg_t == 3 or self.deg_t == 4:
            self.nodes1d = np.array([(1 - np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 - np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 + np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 + np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.])  # Order 8
            self.weights1d = np.array([(18 - np.sqrt(30)) / 72., (18 + np.sqrt(30)) / 72.,
                                       (18 + np.sqrt(30)) / 72., (18 - np.sqrt(30)) / 72.])

    def set_init(self, cfg, dir_mesh, test_case, L0x, Lx, L0t, Lt):

        self.ndim = cfg['ndim']
        self.system_name = cfg['system']
        self.test_case = test_case
        self.save_xt_sol = cfg['save xt sol']
        self.integrator_type = cfg['time integrator']
        self.errType = cfg['error type']
        self.deg_t = cfg['deg_t']

        self.dir_mesh = dir_mesh

        if self.system_name == "waveO1":
            self.deg_x_v = cfg['deg_x_v']
            self.deg_x_sigma = cfg['deg_x_sigma']

            # set polynomial degrees for reference mesh
            try:
                self.ref_lx = cfg["ref xMesh level"]
            except KeyError:
                self.ref_lx = -1

            if self.ref_lx != -1:
                self.deg_ref_x = cfg['deg_x_v'] + 1
                self.deg_ref_x_v = cfg['deg_x_v'] + 1
                self.deg_ref_x_sigma = cfg['deg_x_sigma'] + 1
            else:
                self.ref_lx = -1
                self.deg_ref_x = cfg['deg_x_v'] * 2
                self.deg_ref_x_v = cfg['deg_x_v'] * 2
                self.deg_ref_x_sigma = cfg['deg_x_v'] * 2

        elif self.system_name == "waveO2":
            self.deg_x = cfg['deg_x']

            # set polynomial degrees for reference mesh
            try:
                self.ref_lx = cfg["ref xMesh level"]
            except KeyError:
                self.ref_lx = -1

            if self.ref_lx != -1:
                self.ref_lx = cfg["ref xMesh level"]
                self.deg_ref_x = cfg['deg_x'] + 1
            else:
                self.ref_lx = -1
                self.deg_ref_x = cfg['deg_x'] * 2

        # set combination levels and coefficients
        nSG = 2 * (Lx - L0x) + 1
        self.L = Lx
        self.L0 = L0x
        self.levels = np.zeros((nSG, 2), dtype=np.int32)
        self.comb_coeffs = np.zeros(nSG, dtype=np.float64)

        Ldiff = Lx - L0x
        for k in range(0, Ldiff + 1):
            self.levels[k, 0] = Lx - k
            self.levels[k, 1] = L0t + k
            self.comb_coeffs[k] = 1

        shift = Ldiff + 1
        for k in range(0, Ldiff):
            self.levels[shift + k, 0] = Lx - 1 - k
            self.levels[shift + k, 1] = L0t + k
            self.comb_coeffs[shift + k] = -1

        if self.integrator_type == "Crank-Nicolson":
            self.nne_t = 2
            self.get_time_index_shift = self.get_time_index_shift1
        elif self.integrator_type == "dG":
            self.nne_t = self.deg_t + 1
            self.get_time_index_shift = self.get_time_index_shift2
        else:
            sys.exit("\nUnknown time integrator!\n")

        # choose error computation routine
        if self.save_xt_sol:

            if self.system_name == "waveO1":

                if self.errType == "L2L2":
                    self.compute = self.xtError_L2L2norm_systemO1
                else:
                    sys.exit("\nUnknown xt-error computation!\n")

        else:

            if self.system_name == "waveO1":

                if self.errType == "L2L2":
                    self.compute = self.xError_L2norm_systemO1
                else:
                    sys.exit("\nUnknown x-error computation!\n")

    # get the shift in time index
    def get_time_index_shift1(self, j):
        return j

    def get_time_index_shift2(self, j):
        return j * (self.deg_t + 1)

    # set spatial function spaces and store meshes
    def set(self, xMeshes, tMeshes):

        num_xMeshes = len(xMeshes)
        self.tMeshes = tMeshes

        self.spaces = []
        if self.system_name == "waveO1":

            for k in range(0, num_xMeshes):
                mesh = xMeshes[k]
                W1_elem = df.FiniteElement("DG", mesh.ufl_cell(), self.deg_x_v)
                W2_elem = df.VectorElement("DG", mesh.ufl_cell(), self.deg_x_sigma)
                W = df.FunctionSpace(mesh, df.MixedElement([W1_elem, W2_elem]))
                self.spaces.append(W)

            if self.ndim == 1:
                self.uE = waveO1_locexp.ExactSolution1d(self.test_case.T, self.test_case, degree=self.deg_ref_x)
            elif self.ndim == 2:
                self.uE = waveO1_locexp.ExactSolution2d(self.test_case.T, self.test_case, degree=self.deg_ref_x)
                self.duEdt = waveO1_locexp.ExactSolutionTDeriv2d(self.test_case.T, self.test_case,
                                                                 degree=self.deg_ref_x)
            elif self.ndim == 3:
                self.uE = waveO1_locexp.ExactSolution3d(self.test_case.T, self.test_case, degree=self.deg_ref_x)

            self.xMesh_finest = xMeshes[0]
            # reference function spaces
            if self.ref_lx != -1:
                ref_mesh_file = self.dir_mesh + 'mesh_l%d.h5' % int(self.ref_lx)
                print('\n    Read reference mesh file: ', ref_mesh_file)
                ref_mesh = meshio.read_mesh_h5(ref_mesh_file)
            else:
                ref_mesh = df.refine(df.refine(self.xMesh_finest))

            W1_elem_ref = df.FiniteElement("DG", self.xMesh_finest.ufl_cell(), self.deg_ref_x_v)
            W2_elem_ref = df.VectorElement("DG", self.xMesh_finest.ufl_cell(), self.deg_ref_x_sigma)
            self.W_ref = df.FunctionSpace(ref_mesh, df.MixedElement([W1_elem_ref, W2_elem_ref]))
            print(self.W_ref)

        else:
            sys.exit("\nUnknown system!\n")

    # calls error computation routines
    def eval_error(self, u):
        if self.save_xt_sol:
            print('    Computing space-time error wrt exact solution for sparse xt grid ...\n')
            return self.eval_xtError(u)
        else:
            print('    Computing error wrt exact solution at t=T=%f for sparse xt grid ...\n' % self.test_case.T)
            return self.eval_xError(u)

    # computes sparse-grid error at t=T
    def eval_xError(self, u):

        if self.system_name == "waveO1":
            u_xSolFG = self.build_xSolFG(u)
            errSolV_norm, solV_norm, errSolSigma_norm, solSigma_norm = self.compute(u_xSolFG)
            return np.sqrt(errSolV_norm) / np.sqrt(solV_norm), np.sqrt(errSolSigma_norm) / np.sqrt(solSigma_norm)

        else:
            sys.exit("\nUnknown system!\n")

    # computes space-time sparse-grid error
    def eval_xtError(self, u):

        Ldiff = self.L - self.L0
        nSG = 2 * Ldiff + 1
        self.set_quad_nodes_weights()

        tMesh_finest = self.tMeshes[Ldiff]
        ne_tMesh_finest = tMesh_finest.shape[0] - 1

        if self.system_name == "waveO1":
            errSolV_norm = 0
            solV_norm = 0
            errSolSigma_norm = 0
            solSigma_norm = 0

        for j in range(0, ne_tMesh_finest):

            dt_finest = tMesh_finest[j + 1] - tMesh_finest[j]
            points_t = np.zeros(self.nodes1d.shape[0])
            for k in range(0, self.nodes1d.shape[0]):
                points_t[k] = tMesh_finest[j] + dt_finest * self.nodes1d[k]

            xi_buffer = []
            u_buffer = []
            dt_buffer = []
            for k in range(0, nSG):

                kk = k % (Ldiff + 1)
                tMesh = self.tMeshes[kk]

                factor_t = 2 ** (Ldiff - kk)
                jj = j // factor_t
                tid_shift = self.get_time_index_shift(jj)

                dt_local = dt_finest * factor_t
                dt_buffer.append(dt_local)
                # print(k, dt_local)

                xi_local = np.zeros(self.nodes1d.shape[0])
                for i in range(0, self.nodes1d.shape[0]):
                    xi_local[i] = (points_t[i] - tMesh[jj]) / (tMesh[jj + 1] - tMesh[jj])
                    # print(j, k, i, xi_local[i], points_t[i])

                u_buffer_local = []
                for i in range(0, self.nne_t):
                    u_buffer_local.append(u[k][i + tid_shift])
                    # print(j, k, kk, tid_shift, i+tid_shift)

                xi_buffer.append(xi_local)
                u_buffer.append(u_buffer_local)

            if self.system_name == "waveO1":

                if self.errType == "L2L2":

                    val11, val12, val21, val22 = self.compute(tMesh_finest[j:j + 2], u_buffer, np.asarray(xi_buffer))
                    errSolV_norm += val11
                    solV_norm += val12
                    errSolSigma_norm += val21
                    solSigma_norm += val22
                    print(j, np.sqrt(val11 / val12), np.sqrt(val21 / val22))

            else:
                sys.exit("\nUnknown system!\n")

        if self.system_name == "waveO1":
            return np.sqrt(errSolV_norm) / np.sqrt(solV_norm), np.sqrt(errSolSigma_norm) / np.sqrt(solSigma_norm)
        else:
            sys.exit("\nUnknown system!\n")

    # build full-grid solution, given sparse-grid solution at t=t
    def build_xSolFG(self, u_buffer):

        Ldiff = self.L - self.L0
        nSG = 2 * Ldiff + 1

        space_finest = self.spaces[0]
        u_ = float(self.comb_coeffs[0]) * df.interpolate(u_buffer[0], space_finest)
        for k in range(1, nSG):
            u_ += float(self.comb_coeffs[k]) * df.interpolate(u_buffer[k], space_finest)

        u_FG = df.Function(space_finest)
        u_FG.assign(u_)

        return u_FG

    # build full-grid solution, 
    # given sparse-grid solution for all t \in [t_{n}, t_{n+1}]
    def build_xtSolFG(self, u_buffer, xi_buffer, dt_buffer=[]):

        if self.system_name == "waveO1":
            return self.build_xtSolFG_systemO1(u_buffer, xi_buffer)

    # build full-grid solution,
    # given sparse-grid solution for all t \in [t_{n}, t_{n+1}]
    def build_xtSolFG_systemO1(self, u_buffer, xi_buffer):

        Ldiff = self.L - self.L0
        nSG = 2 * Ldiff + 1

        space_finest = self.spaces[0]

        xi = xi_buffer[0]
        W = self.spaces[0]

        u_k = lagrange.get_sol(self.deg_t, W, u_buffer[0], xi)
        u_ = float(self.comb_coeffs[0]) * df.interpolate(u_k, space_finest)

        for k in range(1, Ldiff + 1):
            kk = k % (Ldiff + 1)
            W = self.spaces[kk]
            xi = xi_buffer[k]
            # print(n, k, kk, xi)

            u_k = lagrange.get_sol(self.deg_t, W, u_buffer[k], xi)
            u_ += float(self.comb_coeffs[k]) * df.interpolate(u_k, space_finest)

        for k in range(Ldiff + 1, nSG):
            kk = k % (Ldiff + 1) + 1
            W = self.spaces[kk]
            xi = xi_buffer[k]
            # print(n, k, kk, xi)

            u_k = lagrange.get_sol(self.deg_t, W, u_buffer[k], xi)
            u_ += float(self.comb_coeffs[k]) * df.interpolate(u_k, space_finest)

        u_FG = df.Function(space_finest)  # full-grid solution
        u_FG.assign(u_)
        # solio.plot_sol2d(u_FG[0])

        return u_FG

    # L2(\Omega \times \{T\}) norm of sparse solution error and exact solution
    def xError_L2norm_systemO1(self, u):
        err_norm = np.zeros(self.ndim + 1)
        uE_norm = np.zeros(self.ndim + 1)

        err_norm[0] = df.assemble(((u[0] - self.uE[0]) ** 2) * df.dx(self.xMesh_finest))
        err_norm[1] = df.assemble(((u[1] - self.uE[1]) ** 2) * df.dx(self.xMesh_finest))
        err_norm[2] = df.assemble(((u[2] - self.uE[2]) ** 2) * df.dx(self.xMesh_finest))

        uE_norm[0] = df.assemble((self.uE[0] ** 2) * df.dx(self.xMesh_finest))
        uE_norm[1] = df.assemble((self.uE[1] ** 2) * df.dx(self.xMesh_finest))
        uE_norm[2] = df.assemble((self.uE[2] ** 2) * df.dx(self.xMesh_finest))

        print(np.sqrt(err_norm) / np.sqrt(uE_norm))
        return err_norm[0], uE_norm[0], np.sum(err_norm[1:3]), np.sum(uE_norm[1:3])

    # L2(tSlab, L2(\Omega)) norm of sparse solution error and exact solution
    def xtError_L2L2norm_systemO1(self, tSlab, u_buffer, xi_buffer):
        err_norm = np.zeros(self.ndim + 1)
        uE_norm = np.zeros(self.ndim + 1)

        for j in range(0, self.nodes1d.shape[0]):

            self.uE.t = tSlab[0] * (1 - self.nodes1d[j]) + tSlab[1] * self.nodes1d[j]
            u = self.build_xtSolFG(u_buffer, xi_buffer[:, j])
            uE_sol = df.interpolate(self.uE, self.W_ref)
            err_fn = df.Function(self.W_ref)
            err_fn.assign(uE_sol - df.interpolate(u, self.W_ref))

            for i in range(0, self.ndim + 1):
                err_norm[i] += self.weights1d[j] * df.assemble((err_fn[i] ** 2) * df.dx(self.xMesh_finest))
                uE_norm[i] += self.weights1d[j] * df.assemble((uE_sol[i] ** 2) * df.dx(self.xMesh_finest))

        err_norm *= (tSlab[1] - tSlab[0])
        uE_norm *= (tSlab[1] - tSlab[0])
        return err_norm[0], uE_norm[0], np.sum(err_norm[1:3]), np.sum(uE_norm[1:3])

# End of file
