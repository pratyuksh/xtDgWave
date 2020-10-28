#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df
from math import *

from pathlib import Path
import sys

cur_dir = str(Path().absolute())  # get working directory
meshes_dir = cur_dir + "/meshes/"  # meshes directory
sols_dir = cur_dir + "/output/"  # solutions directory


# Test cases 2d
def get_test_case_2d(test_case_name):
    if test_case_name == "test1_smooth_unitSquareDomain":
        return Test1SmoothUnitSquareDomain()
    elif test_case_name == "test2_smooth_unitSquareDomain":
        return Test2SmoothUnitSquareDomain()
    elif test_case_name == "test3_smooth_unitSquareDomain":
        return Test3SmoothUnitSquareDomain()
    elif test_case_name == "test4_smooth_unitSquareDomain":
        return Test4SmoothUnitSquareDomain()
    elif test_case_name == "test5_smooth_unitSquareDomain":
        return Test5SmoothUnitSquareDomain()
    elif test_case_name == "test1_singular_gammaShapedDomain":
        return Test1SingularGammaShapedDomain()
    elif test_case_name == "test2_singular_gammaShapedDomain":
        return Test2SingularGammaShapedDomain()
    elif test_case_name == "test1_smooth_squareDomain":
        return Test1SmoothSquareDomain()
    elif test_case_name == "test1_singular_squareDomain":
        return Test1SingularSquareDomain()
    else:
        sys.exit("\nTest case: " + test_case_name + ", not available!\n")


# Smooth solution, Unit Square domain
# u != 0 @ t=0
# source = 0
# Homogeneous Dirichlet
class Test1SmoothUnitSquareDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_quasiUniform/"

        # sound speed
        self.c = 1

        # wave number
        self.kappa = 1

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        uE[0] = +self.kappa * np.sqrt(2) * self.c * np.pi * sin(self.kappa * np.pi * x[0]) * sin(
            self.kappa * np.pi * x[1]) * cos(self.kappa * np.sqrt(2) * pi * self.c * t)
        uE[1] = -self.kappa * np.pi * cos(self.kappa * np.pi * x[0]) * sin(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        uE[2] = -self.kappa * np.pi * sin(self.kappa * np.pi * x[0]) * cos(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        vE = +self.kappa * np.sqrt(2) * self.c * np.pi * sin(self.kappa * np.pi * x[0]) * sin(
            self.kappa * np.pi * x[1]) * cos(self.kappa * np.sqrt(2) * pi * self.c * t)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        sigmaE[0] = -self.kappa * np.pi * cos(self.kappa * np.pi * x[0]) * sin(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        sigmaE[1] = -self.kappa * np.pi * sin(self.kappa * np.pi * x[0]) * cos(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def source(self, t, x):
        f = 0
        return f


# Smooth solution, Unit Square domain
# u != 0 @ t=0
# source = 0
# Homogeneous Dirichlet + Homogeneous Neumann
class Test2SmoothUnitSquareDomain:
    def __init__(self):

        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_quasiUniform/"

        # sound speed
        self.c = 1

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            if (df.near(x[1], 0) or df.near(x[1], 1)) and on_boundary:
                return True
            else:
                return False

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            if (df.near(x[0], 0) or df.near(x[0], 1)) and on_boundary:
                return True
            else:
                return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        uE[0] = -np.sqrt(2) * self.c * np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(np.sqrt(2) * pi * self.c * t)
        uE[1] = +np.pi * sin(np.pi * x[0]) * sin(np.pi * x[1]) * cos(np.sqrt(2) * np.pi * self.c * t)
        uE[2] = -np.pi * cos(np.pi * x[0]) * cos(np.pi * x[1]) * cos(np.sqrt(2) * np.pi * self.c * t)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        duEdt[0] = -2 * (self.c * np.pi) ** 2 * cos(np.pi * x[0]) * sin(np.pi * x[1]) * cos(
            np.sqrt(2) * pi * self.c * t)
        duEdt[1] = -np.sqrt(2) * self.c * (np.pi ** 2) * sin(np.pi * x[0]) * sin(np.pi * x[1]) * sin(
            np.sqrt(2) * np.pi * self.c * t)
        duEdt[2] = +np.sqrt(2) * self.c * (np.pi ** 2) * cos(np.pi * x[0]) * cos(np.pi * x[1]) * sin(
            np.sqrt(2) * np.pi * self.c * t)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        vE = -np.sqrt(2) * self.c * np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(np.sqrt(2) * pi * self.c * t)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        sigmaE[0] = +np.pi * sin(np.pi * x[0]) * sin(np.pi * x[1]) * cos(np.sqrt(2) * np.pi * self.c * t)
        sigmaE[1] = -np.pi * cos(np.pi * x[0]) * cos(np.pi * x[1]) * cos(np.sqrt(2) * np.pi * self.c * t)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def source(self, t, x):
        f = 0
        return f


# Smooth solution, Unit Square domain
# u = 0 @ t=0
# source != 0
# Homogeneous Dirichlet
class Test3SmoothUnitSquareDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_quasiUniform/"

        # sound speed
        self.c = 1

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        uE[0] = 7 * np.pi / 6 * sin(np.pi * x[0]) * sin(np.pi * x[1]) * sin(7 * np.pi / 3 * t)
        uE[1] = -np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(7 * np.pi / 6 * t) ** 2
        uE[2] = -np.pi * sin(np.pi * x[0]) * cos(np.pi * x[1]) * sin(7 * np.pi / 6 * t) ** 2
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        vE = 7 * np.pi / 6 * sin(np.pi * x[0]) * sin(np.pi * x[1]) * sin(7 * np.pi / 3 * t)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        sigmaE[0] = -np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(7 * np.pi / 6 * t) ** 2
        sigmaE[1] = -np.pi * sin(np.pi * x[0]) * cos(np.pi * x[1]) * sin(7 * np.pi / 6 * t) ** 2
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    def source(self, t, x):
        f = sin(np.pi * x[0]) * sin(np.pi * x[1]) * (
                49 * np.pi ** 2 / 18 * cos(7 * np.pi / 3 * t) + 2 * np.pi ** 2 * sin(7 * np.pi / 6 * t) ** 2)
        return f


# Smooth solution, Unit Square domain
# u != 0 @ t=0
# source = 0
# Homogeneous Dirichlet
class Test4SmoothUnitSquareDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_quasiUniform/"

        # sound speed
        self.c = 1

        # wave number
        self.kappa = 1

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        uE[0] = +self.kappa * np.sqrt(2) * self.c * np.pi * sin(self.kappa * np.pi * x[0]) * sin(
            self.kappa * np.pi * x[1]) * cos(self.kappa * np.sqrt(2) * pi * self.c * t)
        uE[1] = -self.kappa * np.pi * cos(self.kappa * np.pi * x[0]) * sin(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        uE[2] = -self.kappa * np.pi * sin(self.kappa * np.pi * x[0]) * cos(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        vE = +self.kappa * np.sqrt(2) * self.c * np.pi * sin(self.kappa * np.pi * x[0]) * sin(
            self.kappa * np.pi * x[1]) * cos(self.kappa * np.sqrt(2) * pi * self.c * t)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        sigmaE[0] = -self.kappa * np.pi * cos(self.kappa * np.pi * x[0]) * sin(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        sigmaE[1] = -self.kappa * np.pi * sin(self.kappa * np.pi * x[0]) * cos(self.kappa * np.pi * x[1]) * sin(
            self.kappa * np.sqrt(2) * np.pi * self.c * t)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def source(self, t, x):
        f = 0
        return f


# Smooth solution, Unit Square domain
# u != 0 @ t=0
# source != 0
# Homogeneous Dirichlet
# Anisotropic medium
class Test5SmoothUnitSquareDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_quasiUniform/"

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        s = np.identity(self.ndim, dtype=np.float64)
        s[0, 1] = 0.5
        s[1, 0] = 0.5
        s[1, 1] = 2
        return s

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        uE[0] = +np.sqrt(2) * np.pi * sin(np.pi * x[0]) * sin(np.pi * x[1]) * cos(np.sqrt(2) * pi * t)
        uE[1] = -np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(np.sqrt(2) * pi * t)
        uE[2] = -np.pi * sin(np.pi * x[0]) * cos(np.pi * x[1]) * sin(np.sqrt(2) * pi * t)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        vE = +np.sqrt(2) * np.pi * sin(np.pi * x[0]) * sin(np.pi * x[1]) * cos(np.sqrt(2) * pi * t)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        sigmaE[0] = -np.pi * cos(np.pi * x[0]) * sin(np.pi * x[1]) * sin(np.sqrt(2) * pi * t)
        sigmaE[1] = -np.pi * sin(np.pi * x[0]) * cos(np.pi * x[1]) * sin(np.sqrt(2) * pi * t)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    def source(self, t, x):
        f = -(np.pi ** 2) * cos(np.pi * (x[0] + x[1])) * sin(np.sqrt(2) * pi * t)
        return f


# Singular solution, Gamma-shaped Domain
# source != 0
class Test1SingularGammaShapedDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1

        self.dir_mesh_quasiUniform = meshes_dir + "gammaShaped_quasiUniform/"
        self.dir_mesh_graded = meshes_dir + "gammaShaped_graded/"
        self.dir_mesh_bisecRefined = meshes_dir + "gammaShaped_bisecRefine/"

        self.dir_sol_quasiUniform = sols_dir + "gammaShaped_quasiUniform/"
        self.dir_sol_graded = sols_dir + "gammaShaped_graded/"
        self.dir_sol_bisecRefined = sols_dir + "gammaShaped_bisecRefine/"

        # sound speed
        self.c = 1

        # solution parameters
        self.gamma = 2. / 3.
        self.kappa = 1
        self.tol = 1e-8

    """
    # Dirichlet boundary
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            if ( (df.near(x[0], 0) and (x[1] <= 0 and x[1] >= -0.5))\
            or (df.near(x[1], 0) and (x[0] <= 0.5 and x[0] >= 0)) ) and on_boundary:
                return True
            else:
                return False
    
    # Neumann boundary
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            if (df.near(x[0],-0.5) or df.near(x[0],+0.5) or df.near(x[1],-0.5) or df.near(x[1],+0.5)) and on_boundary:
                return True
            else:
                return False
    """

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # radial distance from singular vertex
    # noinspection PyMethodMayBeStatic
    def radius_sv(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return r

    # angle wrt edges adjacent to the singular vertex
    # noinspection PyMethodMayBeStatic
    def polar_angle_sv(self, x):
        theta = 2 * np.pi * (x[1] < 0) + atan2(x[1], x[0])
        return theta

    # noinspection PyMethodMayBeStatic
    def pressure(self, t, r, theta):
        vE = self.kappa * np.sqrt(2) * np.pi * (r ** self.gamma) * sin(self.gamma * theta) * cos(
            self.kappa * np.sqrt(2) * np.pi * t)
        return vE

    # noinspection PyMethodMayBeStatic
    def velocity(self, t, r, theta):
        sigmaE = np.zeros(self.ndim)
        if r > self.tol:
            coeff = self.gamma - 1
            sigmaE[0] = -self.gamma * (r ** coeff) * sin(coeff * theta) * sin(self.kappa * np.sqrt(2) * np.pi * t)
            sigmaE[1] = -self.gamma * (r ** coeff) * cos(coeff * theta) * sin(self.kappa * np.sqrt(2) * np.pi * t)
        return sigmaE

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        uE = np.zeros(self.ndim + 1)
        uE[0] = self.pressure(t, r, theta)
        uE[1:self.ndim + 1] = self.velocity(t, r, theta)
        return uE

    # time derivative of exact solution
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        vE = self.pressure(t, r, theta)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        sigmaE = self.velocity(t, r, theta)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    def source(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        f = -((self.kappa * np.sqrt(2) * np.pi) ** 2) * (r ** self.gamma) * sin(self.gamma * theta) * sin(
            self.kappa * np.sqrt(2) * np.pi * t)
        return f


# Singular solution, Gamma-shaped Domain
# source != 0
class Test2SingularGammaShapedDomain:
    def __init__(self):
        self.ndim = 2
        self.T = 1

        self.dir_mesh_quasiUniform = meshes_dir + "gammaShaped_quasiUniform/"
        self.dir_mesh_graded = meshes_dir + "gammaShaped_graded/"
        self.dir_mesh_bisecRefined = meshes_dir + "gammaShaped_bisecRefine/"

        self.dir_sol_quasiUniform = sols_dir + "gammaShaped_quasiUniform/"
        self.dir_sol_graded = sols_dir + "gammaShaped_graded/"
        self.dir_sol_bisecRefined = sols_dir + "gammaShaped_bisecRefine/"

        # sound speed
        self.c = 1

        # solution parameters
        self.gamma = 2. / 3.
        self.kappa = 1
        self.tol = 1e-8

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # medium parameter
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def medium(self, x):
        return self.c

    # medium tensor
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def mediumTensor(self, x):
        return self.c * np.identity(self.ndim, dtype=np.float64)

    # radial distance from singular vertex
    # noinspection PyMethodMayBeStatic
    def radius_sv(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return r

    # angle wrt edges adjacent to the singular vertex
    # noinspection PyMethodMayBeStatic
    def polar_angle_sv(self, x):
        theta = 2 * np.pi * (x[1] < 0) + atan2(x[1], x[0])
        return theta

    # noinspection PyMethodMayBeStatic
    def pressure(self, t, r, theta):
        vE = 7 * np.pi / 6. * self.kappa * (r ** self.gamma) * sin(self.gamma * theta) * sin(
            7 * np.pi / 3. * self.kappa * t)
        return vE

    # noinspection PyMethodMayBeStatic
    def velocity(self, t, r, theta):
        sigmaE = np.zeros(self.ndim)
        if r > self.tol:
            coeff = self.gamma - 1
            sigmaE[0] = -self.gamma * (r ** coeff) * sin(coeff * theta) * sin(7 * np.pi / 6. * self.kappa * t) ** 2
            sigmaE[1] = -self.gamma * (r ** coeff) * cos(coeff * theta) * sin(7 * np.pi / 6. * self.kappa * t) ** 2
        return sigmaE

    # exact solution
    # noinspection PyMethodMayBeStatic
    def exactSol(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        uE = np.zeros(self.ndim + 1)
        uE[0] = self.pressure(t, r, theta)
        uE[1:self.ndim + 1] = self.velocity(t, r, theta)
        return uE

    # time derivative of exact solution
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    def dirichletBC(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        vE = self.pressure(t, r, theta)
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    def neumannBC(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        sigmaE = self.velocity(t, r, theta)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    def source(self, t, x):
        r = self.radius_sv(x)
        theta = self.polar_angle_sv(x)
        f = (49 * np.pi ** 2) / 18. * self.kappa * (r ** self.gamma) * sin(self.gamma * theta) * cos(
            7 * np.pi / 3. * self.kappa * t)

        return f


# Scattering problem
# Smooth solution, square domain [0,2]^2
# u != 0 @ t=0
# source = 0
# Homogeneous Dirichlet
# Heterogeneous medium
class Test1SmoothSquareDomain:
    def __init__(self):

        self.ndim = 2
        self.T = 1
        self.dir_mesh_quasiUniform = meshes_dir + "square_twoPiecewise/quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_twoPiecewise_quasiUniform/"

        self.dir_mesh_bisecRefined = meshes_dir + "square_twoPiecewise/bisecRefine/"
        self.dir_sol_bisecRefined = sols_dir + "square_twoPiecewise_bisecRefine/"

        self.x0 = np.array([1, 1])
        self.xs = 1.20
        self.delta = 0.01

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    def medium(self, x):
        if x[0] <= self.xs:
            return 1.0
        else:
            return 3.0

    # medium tensor
    # noinspection PyMethodMayBeStatic
    def mediumTensor(self, x):
        if x[0] <= self.xs:
            s = 1.0
        else:
            s = 3.0
        return s * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        r = np.sqrt((x[0] - self.x0[0]) ** 2 + (x[1] - self.x0[1]) ** 2)
        uE[1] = np.exp(-(r / self.delta) ** 2) * (2 * (x[0] - self.x0[0]) / self.delta ** 2)
        uE[2] = np.exp(-(r / self.delta) ** 2) * (2 * (x[1] - self.x0[1]) / self.delta ** 2)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def dirichletBC(self, t, x):
        vE = 0
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def source(self, t, x):
        f = 0
        return f


# Transmission problem
# Singular solution, square domain [0,2]^2
# u != 0 @ t=0
# source = 0
# Homogeneous Neumann
# Heterogeneous medium
class Test1SingularSquareDomain:
    def __init__(self):

        self.ndim = 2
        self.T = 0.3

        self.dir_mesh_quasiUniform = meshes_dir + "square_twoPiecewise/quasiUniform/"
        self.dir_sol_quasiUniform = sols_dir + "square_twoPiecewise_quasiUniform/"

        self.dir_mesh_bisecRefined = meshes_dir + "square_twoPiecewise/bisecRefine/"
        self.dir_sol_bisecRefined = sols_dir + "square_twoPiecewise_bisecRefine/"

        self.x0 = np.array([1, 1.125])
        self.xs = 1.20
        self.ys = 1
        self.delta = 0.01

    # Dirichlet boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class DirichletBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Neumann boundary
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    class NeumannBdry(df.SubDomain):
        def inside(self, x, on_boundary):
            return False

    # medium parameter
    # noinspection PyMethodMayBeStatic
    def medium(self, x):
        if x[1] > self.ys and x[0] > self.xs:
            return 3.0
        elif x[1] > self.ys and x[0] <= self.xs:
            return 1.0
        elif x[1] <= self.ys and x[0] <= self.xs:
            return 3.0
        else:
            return 1.0

    # medium tensor
    # noinspection PyMethodMayBeStatic
    def mediumTensor(self, x):
        if x[1] > self.ys and x[0] > self.xs:
            s = 3.0
        elif x[1] > self.ys and x[0] <= self.xs:
            s = 1.0
        elif x[1] <= self.ys and x[0] <= self.xs:
            s = 3.0
        else:
            s = 1.0

        return s * np.identity(self.ndim, dtype=np.float64)

    # exact solution
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol(self, t, x):
        uE = np.zeros(self.ndim + 1)
        r = np.sqrt((x[0] - self.x0[0]) ** 2 + (x[1] - self.x0[1]) ** 2)
        uE[1] = np.exp(-(r / self.delta) ** 2) * (2 * (x[0] - self.x0[0]) / self.delta ** 2)
        uE[2] = np.exp(-(r / self.delta) ** 2) * (2 * (x[1] - self.x0[1]) / self.delta ** 2)
        return uE

    # exact solution time derivative
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def exactSol_tDeriv(self, t, x):
        duEdt = np.zeros(self.ndim + 1)
        return duEdt

    # Dirichlet BC
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def dirichletBC(self, t, x):
        vE = 0
        return vE

    # Neumann BC
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def neumannBC(self, t, x):
        sigmaE = np.zeros(self.ndim)
        return sigmaE

    # source term
    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def source(self, t, x):
        f = 0
        return f

# End of file
