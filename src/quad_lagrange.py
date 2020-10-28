#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import src.lagrange as lagrange
import src.functions as functions


# Assembly: Quadrature bilinear form
class QuadBilin:

    def __init__(self, deg_t):

        self.ndim = 1
        self.deg_t = deg_t

        # Gauss-Legendre nodes and weights, for temporal quadrature
        if deg_t == 1:
            self.nodes1d = np.array([(1 - 1 / np.sqrt(3)) / 2., (1 + 1 / np.sqrt(3)) / 2.])  # Order 4
            self.weights1d = np.array([1. / 2., 1. / 2.])

        elif deg_t == 2:
            self.nodes1d = np.array([(1 - np.sqrt(3. / 5.)) / 2., 0.5, (1 + np.sqrt(3. / 5.)) / 2.])  # Order 6
            self.weights1d = np.array([5. / 18., 4. / 9., 5. / 18.])

        elif deg_t == 3:
            self.nodes1d = np.array([(1 - np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 - np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 + np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     (1 + np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.])  # Order 8
            self.weights1d = np.array([(18 - np.sqrt(30)) / 72., (18 + np.sqrt(30)) / 72., (18 + np.sqrt(30)) / 72.,
                                       (18 - np.sqrt(30)) / 72.])

        elif deg_t == 4:
            self.nodes1d = np.array([(1 - np.sqrt(5 + 2 * np.sqrt(10. / 7.)) / 3.) / 2.,
                                     (1 - np.sqrt(5 - 2 * np.sqrt(10. / 7.)) / 3.) / 2.,
                                     (1 + np.sqrt(5 - 2 * np.sqrt(10. / 7.)) / 3.) / 2.,
                                     (1 + np.sqrt(5 + 2 * np.sqrt(10. / 7.)) / 3.) / 2.,
                                     0.5])  # Order 10
            self.weights1d = np.array(
                [(322 - 13 * np.sqrt(70)) / 1800., (322 + 13 * np.sqrt(70)) / 1800., (322 + 13 * np.sqrt(70)) / 1800.,
                 (322 - 13 * np.sqrt(70)) / 1800., 64. / 225.])

    # Quadrature: _temporal_ Mass form on the reference interval [0,1]
    def mass_temporal_refEl(self, pt, id_basis_t1, id_basis_t2):

        val_t = 0
        for k in range(0, self.nodes1d.shape[0]):
            basis_t1 = lagrange.basis1d(pt, id_basis_t1, self.nodes1d[k])
            basis_t2 = lagrange.basis1d(pt, id_basis_t2, self.nodes1d[k])
            val_t += self.weights1d[k] * basis_t1 * basis_t2

        return val_t

    # Quadrature: _temporal_ L2-projection on degree 'p-1' polynomial space
    def projection_temporal_refEl(self, pt, id_basis_t1, id_basis_t2):

        val_t = 0
        for k in range(0, self.nodes1d.shape[0]):
            basis_t1 = lagrange.basis1d(pt - 1, id_basis_t1, self.nodes1d[k])
            basis_t2 = lagrange.basis1d(pt, id_basis_t2, self.nodes1d[k])
            val_t += self.weights1d[k] * basis_t1 * basis_t2

        return val_t

    # Quadrature: _temporal_ Stiffness form on reference interval [0,1]
    def stiffness_temporal_refEl(self, pt, id_basis_t1, id_basis_t2):

        val_t = 0
        for k in range(0, self.nodes1d.shape[0]):
            basisDeriv_t1 = lagrange.basis1d_deriv(pt, id_basis_t1, self.nodes1d[k])
            basisDeriv_t2 = lagrange.basis1d_deriv(pt, id_basis_t2, self.nodes1d[k])
            val_t += self.weights1d[k] * basisDeriv_t1 * basisDeriv_t2

        return val_t


# Assembly: Quadrature rhs linear functional
class QuadRhs:

    def __init__(self, deg_t):

        self.ndim = 1
        self.deg_t = deg_t

        # Gauss-Legendre nodes and weights, for temporal quadrature
        if self.deg_t == 1:

            self.nodes1d = np.array([(1 - 1 / np.sqrt(3)) / 2., (1 + 1 / np.sqrt(3)) / 2.])  # Degree 3
            self.weights1d = np.array([1. / 2., 1. / 2.])

        elif self.deg_t == 2:

            self.nodes1d = np.array([(1 - np.sqrt(3. / 5.)) / 2., 1. / 2., (1 + np.sqrt(3. / 5.)) / 2.])  # Degree 5
            self.weights1d = np.array([5. / 18., 4. / 9., 5. / 18.])

        elif self.deg_t == 3:

            self.nodes1d = np.array([+(1 - np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     +(1 - np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     +(1 + np.sqrt(3. / 7. - 2. / 7. * np.sqrt(6. / 5.))) / 2.,
                                     +(1 + np.sqrt(3. / 7. + 2. / 7. * np.sqrt(6. / 5.))) / 2.])

            self.weights1d = np.array([(18. - np.sqrt(30)) / 72., (18. + np.sqrt(30)) / 72., (18. + np.sqrt(30)) / 72.,
                                       (18. - np.sqrt(30)) / 72.])

    # Quadrature formula for \int_\hat{I_t} (f(t)*basis_t) d\hat{t}
    def projectionL2_local(self, ndof_x, f, a, b):

        p = self.deg_t
        nq = self.nodes1d.shape[0]

        f_nodes = []
        t_tilde = np.zeros(nq)
        for k in range(0, nq):
            t_tilde[k] = functions.affineMap(a, b, self.nodes1d[k])
            f_nodes.append(f(t_tilde[k]).get_local())

        val = np.zeros(p * ndof_x)
        for k in range(0, p):
            for q in range(0, nq):
                basisVal_t = lagrange.basis1d(p, k, self.nodes1d[q])
                val[k * ndof_x: (k + 1) * ndof_x] += self.weights1d[q] * f_nodes[q] * basisVal_t

        return val * (b - a)

    # Quadrature formula for \int_\hat{I_t} (f(t)*basis_t) d\hat{t}
    def projectionL2(self, ndof_x, f, t_nm2, t_nm1, t_n):

        p = self.deg_t
        nq = self.nodes1d.shape[0]

        f_nodes = []
        t_tilde = np.zeros(nq)
        for k in range(0, nq):
            t_tilde[k] = functions.affineMap(t_nm2, t_nm1, self.nodes1d[k])
            f_nodes.append(f(t_tilde[k]).get_local())

        val_tmp = np.zeros(ndof_x)
        for q in range(0, nq):
            basisVal_t = lagrange.basis1d(p, p, self.nodes1d[q])
            val_tmp[0:ndof_x] += self.weights1d[q] * f_nodes[q] * basisVal_t

        val = self.projectionL2_local(ndof_x, f, t_nm1, t_n)
        val[0:ndof_x] += (t_nm1 - t_nm2) * val_tmp[0:ndof_x]

        return val


# End of file
