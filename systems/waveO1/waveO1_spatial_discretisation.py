#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as df
import systems.waveO1.local_expressions as locexp


# First-order Wave equation: spatial dG discretisation
class WaveSpatialDG:

    def __init__(self):
        self.deg_v = None
        self.deg_sigma = None
        self.deg_ref = None
        self.ndim = None
        self.nvars = None

        self.init_sol = None
        self.source = None
        self.dirichletBC = None
        self.neumannBC = None

        self.mesh = None
        self.alphaInt = None
        self.betaInt = None
        self.alphaBnd = None
        self.betaBnd = None

        self.W1_elem = None
        self.W2_elem = None
        self.W = None
        self.w = None
        self.tau = None

        self.FunctionSpace = None
        self.FunctionSpace_fine = None

        self.normal = None
        self.ds = None
        self.mediumPar = None
        self.mediumTensor = None

    def set_system(self, cfg, test_case, mesh, dt=1):

        self.deg_v = cfg['deg_x_v']
        self.deg_sigma = cfg['deg_x_sigma']

        self.deg_ref = self.deg_v + 2
        self.ndim = test_case.ndim
        self.nvars = self.ndim + 1

        # initial conditions
        if self.ndim == 1:
            self.init_sol = locexp.ExactSolution1d(0, test_case, degree=self.deg_ref)
        elif self.ndim == 2:
            self.init_sol = locexp.ExactSolution2d(0, test_case, degree=self.deg_ref)

        # source
        self.source = locexp.Source(0, test_case, degree=self.deg_ref)

        # Dirichlet BC
        self.dirichletBC = locexp.DirichletBc(0, test_case, degree=self.deg_ref)

        # Neumann BC
        if self.ndim == 1:
            self.neumannBC = locexp.NeumannBc1d(0, test_case, degree=self.deg_ref)
        elif self.ndim == 2:
            self.neumannBC = locexp.NeumannBc2d(0, test_case, degree=self.deg_ref)

        # mark Dirichlet and Neumann boundaries
        boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        test_case.DirichletBdry().mark(boundaries, 1)
        test_case.NeumannBdry().mark(boundaries, 2)

        # mesh and parameters
        self.mesh = mesh
        hF = df.FacetArea(mesh)

        # choice of stabilization parameters
        try:
            stabParamsType = cfg["stab params type"]
        except KeyError:
            stabParamsType = 1  # cases 1,2,3,4,5
        print("Stab Params case: ", stabParamsType)
        if stabParamsType == 1:
            self.betaInt = df.Constant('1.0')
            self.alphaInt = df.Constant('1.0')

            self.betaBnd = df.Constant('1.0')
            self.alphaBnd = df.Constant('1.0')

        elif stabParamsType == 2:
            self.betaInt = df.Constant('1.0')
            self.alphaInt = 2. / (hF('+') + hF('-'))

            self.betaBnd = df.Constant('1.0')
            self.alphaBnd = 1. / hF('+')

        elif stabParamsType == 3:
            self.betaInt = 0.5 * (hF('+') + hF('-'))
            self.alphaInt = df.Constant('1.0')

            self.betaBnd = hF('+')
            self.alphaBnd = df.Constant('1.0')

        elif stabParamsType == 4:
            self.betaInt = 0.5 * (hF('+') + hF('-'))
            self.alphaInt = 2. / (hF('+') + hF('-'))

            self.betaBnd = hF('+')
            self.alphaBnd = 1. / hF('+')

        elif stabParamsType == 5:
            self.betaInt = 0.5 * (hF('+') + hF('-')) / dt
            self.alphaInt = 2 * dt / (hF('+') + hF('-'))

            self.betaBnd = hF('+') / dt
            self.alphaBnd = dt / hF('+')

        elif stabParamsType == 6:
            self.betaInt = (0.5 * (hF('+') + hF('-'))) ** 0.5
            self.alphaInt = (2. / (hF('+') + hF('-'))) ** 0.5

            self.betaBnd = (hF('+')) ** 0.5
            self.alphaBnd = 1. / (hF('+')) ** 0.5

        # test space
        self.W1_elem = df.FiniteElement("DG", mesh.ufl_cell(), self.deg_v)
        self.W2_elem = df.VectorElement("DG", mesh.ufl_cell(), self.deg_sigma)
        self.W = df.FunctionSpace(mesh, df.MixedElement([self.W1_elem, self.W2_elem]))
        (self.w, self.tau) = df.TestFunctions(self.W)
        self.FunctionSpace = self.W  # for external referencing
        print(self.W)

        # upsampling for visualization
        mesh_fine = df.refine(df.refine(df.refine(mesh)))
        W_fine = df.FunctionSpace(mesh_fine, df.MixedElement([self.W1_elem, self.W2_elem]))
        self.FunctionSpace_fine = W_fine

        # facet normals
        self.normal = df.FacetNormal(mesh)

        # measure at boundaries
        self.ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)

        # medium parameters
        self.mediumTensor = locexp.MediumAniso(test_case, degree=self.deg_ref, domain=self.mesh)
        self.mediumPar = locexp.MediumIso(test_case, degree=self.deg_ref)


    # assemble system matrices
    # isotropic medium
    def assemble_system(self):

        (v, sigma) = df.TrialFunctions(self.W)
        (w, tau) = df.TestFunctions(self.W)

        # mass
        mass_form = v * w / (self.mediumPar ** 2) * df.dx + df.inner(sigma, tau) * df.dx
        mass = df.assemble(mass_form)

        # convection
        convec_form = df.div(sigma) * w * df.dx + df.inner(df.grad(v), tau) * df.dx
        convec = df.assemble(convec_form)

        # upwind numerical flux
        flux_form = (df.jump(v) * df.inner(self.normal('-'), df.avg(tau)) + df.inner(self.normal('-'),
                                                                                     df.jump(sigma)) * df.avg(
            w)) * df.dS
        flux_form += (self.alphaInt * df.jump(v) * df.jump(w) + self.betaInt * df.inner(self.normal('-'),
                                                                                        df.jump(sigma)) * df.inner(
            self.normal('-'), df.jump(tau))) * df.dS
        flux = df.assemble(flux_form)

        # Dirichlet and Neumann boundary
        dirichlet_bdry_form = (-v * df.inner(self.normal, tau) + self.alphaBnd * v * w) * self.ds(1)
        dirichlet_bdry = df.assemble(dirichlet_bdry_form)

        neumann_bdry_form = (-w * df.inner(self.normal, sigma) + self.betaBnd * df.inner(self.normal, sigma) * df.inner(
            self.normal, tau)) * self.ds(2)
        neumann_bdry = df.assemble(neumann_bdry_form)

        # stiffness matrix
        stiff = convec + flux + dirichlet_bdry + neumann_bdry

        return mass, stiff

    # assemble system matrices
    # implement the discrete formulation from XT-DG manuscript
    # equivalent as the one above
    def assemble_system_test(self):

        (v, sigma) = df.TrialFunctions(self.W)
        (w, tau) = df.TestFunctions(self.W)

        # mass
        mass_form = v * w / (self.mediumPar ** 2) * df.dx + df.inner(sigma, tau) * df.dx
        mass = df.assemble(mass_form)

        # convection
        convec_form = - df.div(tau) * v * df.dx - df.inner(df.grad(w), sigma) * df.dx
        convec = df.assemble(convec_form)

        # upwind numerical flux
        flux_form = (df.avg(v) * df.inner(self.normal('+'), df.jump(tau)) + df.jump(w) * df.inner(self.normal('+'),
                                                                                                  df.avg(
                                                                                                      sigma))) * df.dS
        flux_form += (self.alphaInt * df.jump(v) * df.jump(w) + self.betaInt * df.inner(self.normal('+'),
                                                                                        df.jump(sigma)) * df.inner(
            self.normal('+'), df.jump(tau))) * df.dS
        flux = df.assemble(flux_form)

        # Dirichlet and Neumann boundary
        dirichlet_bdry_form = (w * df.inner(self.normal, sigma) + self.alphaBnd * v * w) * self.ds(1)
        dirichlet_bdry = df.assemble(dirichlet_bdry_form)

        neumann_bdry_form = (v * df.inner(self.normal, tau) + self.betaBnd * df.inner(self.normal, sigma) * df.inner(
            self.normal, tau)) * self.ds(2)
        neumann_bdry = df.assemble(neumann_bdry_form)

        # stiffness matrix
        stiff = convec + flux + dirichlet_bdry + neumann_bdry

        return mass, stiff

    # assemble system matrices
    def assemble_system_aniso(self):

        (v, sigma) = df.TrialFunctions(self.W)
        (w, tau) = df.TestFunctions(self.W)

        # mass
        mass_form = v * w * df.dx + df.inner(sigma, tau) * df.dx
        mass = df.assemble(mass_form)

        # convection
        convec_form = df.div(self.mediumTensor * sigma) * w * df.dx + df.inner(df.grad(v), tau) * df.dx
        convec = df.assemble(convec_form)

        # upwind numerical flux
        flux_form = (df.jump(v) * df.inner(self.normal('-'), df.avg(tau)) + df.inner(self.normal('-'),
                                                                                     self.mediumTensor * df.jump(
                                                                                         sigma)) * df.avg(w)) * df.dS
        flux_form += (self.alphaInt * df.jump(v) * df.jump(w) + self.betaInt * df.inner(self.normal('-'),
                                                                                        df.jump(sigma)) * df.inner(
            self.normal('-'), df.jump(tau))) * df.dS
        flux = df.assemble(flux_form)

        # Dirichlet and Neumann boundary
        dirichlet_bdry_form = (-v * df.inner(self.normal, tau) + self.alphaBnd * v * w) * self.ds(1)
        dirichlet_bdry = df.assemble(dirichlet_bdry_form)

        neumann_bdry_form = (-w * df.inner(self.normal, self.mediumTensor * sigma) + self.betaBnd * df.inner(
            self.normal, sigma) * df.inner(self.normal, tau)) * self.ds(2)
        neumann_bdry = df.assemble(neumann_bdry_form)

        # stiffness matrix
        stiff = convec + flux + dirichlet_bdry + neumann_bdry

        return mass, stiff

    # assemble rhs
    def assemble_rhs(self, t):
        return self.assemble_source(t) + self.assemble_BCs(t)

    # assemble source
    def assemble_source(self, t):
        self.source.t = t
        source_fun = self.source * self.w * df.dx
        return df.assemble(source_fun)

    # assemble Dirichlet and Neumann BCs
    def assemble_BCs(self, t):

        self.dirichletBC.t = t
        self.neumannBC.t = t

        dirichletBC_fun = self.dirichletBC * (self.alphaBnd * self.w - df.inner(self.normal, self.tau)) * self.ds(1)
        neumannBC_fun = None
        if self.ndim == 1:
            neumannBC_fun = self.normal[0] * self.neumannBC * (
                        self.betaBnd * df.inner(self.normal, self.tau) - self.w) * self.ds(2)
        elif self.ndim == 2:
            # neumannBC_fun = -df.inner(self.normal, self.mediumTensor*self.neumannBC)*self.w*self.ds(2)\
            # +self.betaBnd*df.inner(self.normal, self.neumannBC)*df.inner(self.normal, self.tau)*self.ds(2)
            neumannBC_fun = df.inner(self.normal, self.neumannBC) * (
                        self.betaBnd * df.inner(self.normal, self.tau) - self.w) * self.ds(2)

        return df.assemble(dirichletBC_fun) + df.assemble(neumannBC_fun)

    # apply Dirichlet BC in strong form
    def apply_dirichletBCs(self, S, t):
        pass

# End of file
