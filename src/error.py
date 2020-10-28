#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df
import sys

import src.lagrange as lagrange
import systems.waveO1.local_expressions as waveO1_locexp
import mesh_generator.io_mesh as meshio


class Error:

    def __init__(self, cfg, dir_mesh, test_case):

        self.ndim = cfg['ndim']
        self.system_name = cfg['system']
        self.test_case = test_case
        self.save_xt_sol = cfg['save xt sol']
        self.integrator_type = cfg['time integrator']
        self.errType = cfg['error type']

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

        self.deg_t = cfg['deg_t']

        # choice of stabilization parameters
        try:
            self.stabParamsType = cfg["stab params type"]
        except KeyError:
            self.stabParamsType = 1  # cases 1,2,3,4,5

        self.dir_mesh = dir_mesh
        self.mesh = None
        self.ref_mesh = None
        self.uE = None
        self.duEdt = None
        self.W = None
        self.W_ref = None
        self.compute = None

        self.nodes1d = None
        self.weights1d = None
        self.set_quad_nodes_weights()

        self.mediumPar = None

        self.hF = None
        self.normal = None
        self.ds = None

        self.betaInt = None
        self.alphaInt = None
        self.betaBnd = None
        self.alphaBnd = None

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

    def set_system(self, mesh):

        self.mesh = mesh

        if self.system_name == "waveO1":

            if self.ndim == 1:
                self.uE = waveO1_locexp.ExactSolution1d(self.test_case.T, self.test_case, degree=self.deg_ref_x)
            elif self.ndim == 2:
                self.uE = waveO1_locexp.ExactSolution2d(self.test_case.T, self.test_case, degree=self.deg_ref_x)
                self.duEdt = waveO1_locexp.ExactSolutionTDeriv2d(self.test_case.T, self.test_case,
                                                                 degree=self.deg_ref_x)
            elif self.ndim == 3:
                self.uE = waveO1_locexp.ExactSolution3d(self.test_case.T, self.test_case, degree=self.deg_ref_x)

            W1_elem = df.FiniteElement("DG", mesh.ufl_cell(), self.deg_x_v)
            W2_elem = df.VectorElement("DG", mesh.ufl_cell(), self.deg_x_sigma)
            self.W = df.FunctionSpace(mesh, df.MixedElement([W1_elem, W2_elem]))

            # reference function spaces
            if self.ref_lx != -1:
                ref_mesh_file = self.dir_mesh + 'mesh_l%d.h5' % self.ref_lx
                print('\n    Read reference mesh file: ', ref_mesh_file)
                ref_mesh = meshio.read_mesh_h5(ref_mesh_file)
            else:
                ref_mesh = df.refine(df.refine(mesh))
            self.ref_mesh = ref_mesh

            W1_elem_ref = df.FiniteElement("DG", mesh.ufl_cell(), self.deg_ref_x_v)
            W2_elem_ref = df.VectorElement("DG", mesh.ufl_cell(), self.deg_ref_x_sigma)
            self.W_ref = df.FunctionSpace(ref_mesh, df.MixedElement([W1_elem_ref, W2_elem_ref]))
            print(self.W_ref)

            # mark Dirichlet and Neumann boundaries
            boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
            boundaries.set_all(0)
            self.test_case.DirichletBdry().mark(boundaries, 1)
            self.test_case.NeumannBdry().mark(boundaries, 2)

            self.mediumPar = waveO1_locexp.MediumIso(self.test_case, degree=self.deg_ref_x)

            # mesh facets
            self.hF = df.FacetArea(mesh)

            # facet normals
            self.normal = df.FacetNormal(mesh)

            # measure at boundaries
            self.ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)

            # stabilization parameters
            if self.stabParamsType == 1:
                self.betaInt = df.Constant('1.0')
                self.alphaInt = df.Constant('1.0')

                self.betaBnd = df.Constant('1.0')
                self.alphaBnd = df.Constant('1.0')

            elif self.stabParamsType == 2:
                self.betaInt = df.Constant('1.0')
                self.alphaInt = 2. / (self.hF('+') + self.hF('-'))

                self.betaBnd = df.Constant('1.0')
                self.alphaBnd = 1. / self.hF('+')

            elif self.stabParamsType == 3:
                self.betaInt = 0.5 * (self.hF('+') + self.hF('-'))
                self.alphaInt = df.Constant('1.0')

                self.betaBnd = self.hF('+')
                self.alphaBnd = df.Constant('1.0')

            elif self.stabParamsType == 4:
                self.betaInt = 0.5 * (self.hF('+') + self.hF('-'))
                self.alphaInt = 2. / (self.hF('+') + self.hF('-'))

                self.betaBnd = self.hF('+')
                self.alphaBnd = 1. / self.hF('+')

        else:
            sys.exit("\nUnknown system!\n")

        # choose error computation routine
        if self.save_xt_sol:

            if self.system_name == "waveO1":

                if self.errType == "L2L2":
                    self.compute = self.xtError_L2L2norm_systemO1
                elif self.errType == "DG":
                    self.compute = self.xtError_DGnormFtime_systemO1
                else:
                    sys.exit("\nUnknown xt-error computation!\n")
            
        else:

            if self.system_name == "waveO1":

                if self.errType == "L2L2":
                    self.compute = self.xError_L2norm_systemO1
                else:
                    sys.exit("\nUnknown x-error computation!\n")

    def eval(self, t, u):

        if self.system_name == "waveO1":
            return self.eval_error_systemO1(t, u)

    def eval_error_systemO1(self, t, u):

        ne_t = t.shape[0] - 1

        errSolV_norm = 0
        solV_norm = 0

        errSolSigma_norm = 0
        solSigma_norm = 0

        if self.save_xt_sol:

            print('    Computing space-time error wrt exact solution ...\n')

            if self.integrator_type == "Crank-Nicolson":

                for j in range(0, ne_t):

                    # collect the vertex coordinates of an element
                    tSlab = np.array([t[j], t[j + 1]])

                    if self.errType == "L2L2":

                        val11, val12, val21, val22 = self.compute(tSlab, u[j:j + 2])
                        errSolV_norm += val11
                        solV_norm += val12
                        errSolSigma_norm += val21
                        solSigma_norm += val22
                        print(j, np.sqrt(val11 / val12), np.sqrt(val21 / val22))

            elif self.integrator_type == "dG":

                # compute the DG norm
                if self.errType == "DG":
                    dgErrSolFtime_norm = 0
                    dgErrSolFspace_norm = 0

                    # time-like faces
                    for j in range(0, ne_t):
                        tSlab = np.array([t[j], t[j + 1]])
                        val = self.xtError_DGnormFtime_systemO1(tSlab,
                                                                u[j * (self.deg_t + 1): (j + 1) * (self.deg_t + 1)])
                        dgErrSolFtime_norm += val
                        del val
                        del tSlab

                    # space-like faces
                    for j in range(1, ne_t):
                        uColl = [u[(j - 1) * (self.deg_t + 1): j * (self.deg_t + 1)],
                                 u[j * (self.deg_t + 1): (j + 1) * (self.deg_t + 1)]]
                        val = self.xtError_DGnormFspace_systemO1(t[j], uColl)
                        dgErrSolFspace_norm += val
                        del val
                        del uColl

                    # @t=0 and t=T
                    val1 = self.xtError_DGnormFspace_systemO1(t[0], [u[0: (self.deg_t + 1)]])
                    val2 = self.xtError_DGnormFspace_systemO1(t[ne_t], [u[(ne_t - 1) * (self.deg_t + 1):
                                                                          ne_t * (self.deg_t + 1)]])
                    dgErrSolFspace_norm += val1 + val2
                    dgErrSolFspace_norm *= 0.5

                    print('DG err norm: ', np.sqrt(dgErrSolFtime_norm), np.sqrt(dgErrSolFspace_norm))
                    return np.sqrt(dgErrSolFspace_norm), np.sqrt(dgErrSolFtime_norm)

                for j in range(0, ne_t):

                    # collect the vertex coordinates of an element
                    tSlab = np.array([t[j], t[j + 1]])

                    if self.errType == "L2L2":

                        val11, val12, val21, val22 = self.compute(tSlab,
                                                                  u[j * (self.deg_t + 1): (j + 1) * (self.deg_t + 1)])
                        errSolV_norm += val11
                        solV_norm += val12
                        errSolSigma_norm += val21
                        solSigma_norm += val22
                        print(j, np.sqrt(val11 / val12), np.sqrt(val21 / val22))

            else:
                sys.exit("\nUnknown time integrator!\n")

        else:

            print('    Computing error wrt exact solution at t=T=%f ...\n' % self.test_case.T)
            errSolV_norm, solV_norm, errSolSigma_norm, solSigma_norm = self.compute(u)

        return np.sqrt(errSolV_norm) / np.sqrt(solV_norm), np.sqrt(errSolSigma_norm) / np.sqrt(solSigma_norm)


    # L2(\Omega \times \{T\}) norm of solution error and exact solution
    def xError_L2norm_systemO1(self, u):
        err_norm = np.zeros(self.ndim + 1)
        uE_norm = np.zeros(self.ndim + 1)

        err_norm[0] = df.assemble(((u[0] - self.uE[0]) ** 2) * df.dx(self.mesh))
        err_norm[1] = df.assemble(((u[1] - self.uE[1]) ** 2) * df.dx(self.mesh))
        err_norm[2] = df.assemble(((u[2] - self.uE[2]) ** 2) * df.dx(self.mesh))

        uE_norm[0] = df.assemble((self.uE[0] ** 2) * df.dx(self.mesh))
        uE_norm[1] = df.assemble((self.uE[1] ** 2) * df.dx(self.mesh))
        uE_norm[2] = df.assemble((self.uE[2] ** 2) * df.dx(self.mesh))

        return err_norm[0], uE_norm[0], np.sum(err_norm[1:3]), np.sum(uE_norm[1:3])

    # L2(tSlab, L2(\Omega)) norm of solution error and exact solution
    def xtError_L2L2norm_systemO1(self, tSlab, u):
        err_norm = np.zeros(self.ndim + 1)
        uE_norm = np.zeros(self.ndim + 1)

        for j in range(0, self.nodes1d.shape[0]):

            t = tSlab[0] * (1 - self.nodes1d[j]) + tSlab[1] * self.nodes1d[j]
            u_ = lagrange.get_sol(self.deg_t, self.W, u, self.nodes1d[j])
            self.uE.t = t

            uE_sol = df.interpolate(self.uE, self.W_ref)
            err_fn = df.Function(self.W_ref)
            err_fn.assign(uE_sol - df.interpolate(u_, self.W_ref))

            for i in range(0, self.ndim + 1):
                err_norm[i] += self.weights1d[j] * df.assemble((err_fn[i] ** 2) * df.dx(self.mesh))
                uE_norm[i] += self.weights1d[j] * df.assemble((uE_sol[i] ** 2) * df.dx(self.mesh))

        err_norm *= (tSlab[1] - tSlab[0])
        uE_norm *= (tSlab[1] - tSlab[0])
        return err_norm[0], uE_norm[0], np.sum(err_norm[1:3]), np.sum(uE_norm[1:3])


    # DG error norm of solution
    def xtError_DGnormFtime_systemO1(self, tSlab, u):
        err_vJump = 0
        err_vD = 0

        err_sigmaJump = 0
        err_sigmaN = 0

        for j in range(0, self.nodes1d.shape[0]):
            t = tSlab[0] * (1 - self.nodes1d[j]) + tSlab[1] * self.nodes1d[j]
            u_ = lagrange.get_sol(self.deg_t, self.W, u, self.nodes1d[j])
            self.uE.t = t

            uE_sol = df.interpolate(self.uE, self.W)
            err_fn = df.Function(self.W)
            err_fn.assign(uE_sol - df.interpolate(u_, self.W))
            (err_vFn, err_sigmaFn) = err_fn.split()

            # error in v at time-like faces
            err_vJump += self.weights1d[j] * df.assemble(self.alphaInt * (df.jump(err_vFn) ** 2) * df.dS(self.mesh))
            err_vD += self.weights1d[j] * df.assemble(self.alphaBnd * (err_vFn ** 2) * df.ds(1))
            # error in sigma at time-like faces
            err_sigmaJump += self.weights1d[j] * df.assemble(self.betaInt * (df.inner(self.normal('-'),
                                                                                      df.jump(err_sigmaFn)) ** 2) *
                                                             df.dS(self.mesh))
            err_sigmaN += self.weights1d[j] * df.assemble(self.betaBnd * (df.inner(self.normal, err_sigmaFn) ** 2) *
                                                          df.ds(2))

        err_norm = err_vJump + err_sigmaJump + err_vD + err_sigmaN
        # print(err_vJump, err_vD, err_sigmaJump, err_sigmaN)

        err_norm *= (tSlab[1] - tSlab[0])
        return err_norm

    def xtError_DGnormFspace_systemO1(self, tn, uColl):
        TOL = 1E-6
        if tn <= TOL:  # @t=0
            up = lagrange.get_sol(self.deg_t, self.W, uColl[0], 0)
            self.uE.t = 0
            err_fn = df.Function(self.W)
            err_fn.assign(df.interpolate(self.uE, self.W) - df.interpolate(up, self.W))

        elif self.test_case.T - tn <= TOL:  # @t=T
            um = lagrange.get_sol(self.deg_t, self.W, uColl[0], 1)
            self.uE.t = self.test_case.T
            err_fn = df.Function(self.W)
            err_fn.assign(df.interpolate(self.uE, self.W) - df.interpolate(um, self.W))

        else:
            um = lagrange.get_sol(self.deg_t, self.W, uColl[0], 1)
            up = lagrange.get_sol(self.deg_t, self.W, uColl[1], 0)
            err_fn = df.Function(self.W)
            err_fn.assign(df.interpolate(up, self.W) - df.interpolate(um, self.W))

        err_vJump = df.assemble(((err_fn.sub(0) / self.mediumPar) ** 2) * df.dx(self.mesh))
        err_sigmaJump = df.assemble((err_fn.sub(1) ** 2) * df.dx(self.mesh))
        err_norm = err_vJump + err_sigmaJump
        # print(err_vJump, err_sigmaJump)
        return err_norm

# End of file
