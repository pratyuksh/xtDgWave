#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as df


# FEniCS Expression: 1d exact solution
class ExactSolution1d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        uE = self.test_case.exactSol(self.t, x)
        values[0] = uE[0]
        values[1] = uE[1]

    def value_shape(self):
        return (2,)


# FEniCS Expression: time derivative of 1d exact solution
class ExactSolutionTDeriv1d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        duEdt = self.test_case.exactSol_tDeriv(self.t, x)
        values[0] = duEdt[0]
        values[1] = duEdt[1]

    def value_shape(self):
        return (2,)


# FEniCS Expression: 2d exact solution
class ExactSolution2d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        uE = self.test_case.exactSol(self.t, x)
        values[0] = uE[0]
        values[1] = uE[1]
        values[2] = uE[2]

    def value_shape(self):
        return (3,)


# FEniCS Expression: time derivative of 2d exact solution
class ExactSolutionTDeriv2d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        duEdt = self.test_case.exactSol_tDeriv(self.t, x)
        values[0] = duEdt[0]
        values[1] = duEdt[1]
        values[2] = duEdt[2]

    def value_shape(self):
        return (3,)


# FEniCS Expression: 3d exact solution
class ExactSolution3d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        uE = self.test_case.exactSol(self.t, x)
        values[0] = uE[0]
        values[1] = uE[1]
        values[2] = uE[2]
        values[3] = uE[3]

    def value_shape(self):
        return (4,)


# FEniCS Expression: 1d Neumann BC
class NeumannBc1d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        values[0] = self.test_case.neumannBC(self.t, x)

    def value_shape(self):
        return ()


# FEniCS Expression: 2d Neumann BC
class NeumannBc2d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        sigmaE = self.test_case.neumannBC(self.t, x)
        values[0] = sigmaE[0]
        values[1] = sigmaE[1]

    def value_shape(self):
        return (2,)


# FEniCS Expression: 3d Neumann BC
class NeumannBc3d(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        sigmaE = self.test_case.neumannBC(self.t, x)
        values[0] = sigmaE[0]
        values[1] = sigmaE[1]
        values[2] = sigmaE[2]

    def value_shape(self):
        return (3,)


# FEniCS Expression: 1d/2d Dirichlet BC
class DirichletBc(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        values[0] = self.test_case.dirichletBC(self.t, x)

    def value_shape(self):
        return ()


# FEniCS Expression: 1d/2d Source
class Source(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        values[0] = self.test_case.source(self.t, x)

    def value_shape(self):
        return ()


# FEniCS Expression: 1d/2d isotropic medium
class MediumIso(df.Expression):

    def __init__(self, test_case, **kwargs):
        self.test_case = test_case

    def eval(self, values, x):
        values[0] = self.test_case.medium(x)

    def value_shape(self):
        return ()


# FEniCS Expression: 1d/2d isotropic medium time-dependent
class MediumIsoTD(df.Expression):

    def __init__(self, t, test_case, **kwargs):
        self.t = t
        self.test_case = test_case

    def eval(self, values, x):
        values[0] = self.test_case.medium(self.t, x)

    def value_shape(self):
        return ()


# FEniCS Expression: 2d anisotropic medium
class MediumAniso(df.Expression):

    def __init__(self, test_case, **kwargs):
        self.test_case = test_case

    def eval(self, values, x):
        medium = self.test_case.mediumTensor(x)
        values[0] = medium[0, 0]
        values[1] = medium[0, 1]
        values[2] = medium[1, 0]
        values[3] = medium[1, 1]

    def value_shape(self):
        return (2, 2)

# End of file
