#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import src.legendre as legendre
import src.functions as functions


class Quad:

    def __init__(self, deg_t):

        self.deg_t = deg_t

        if deg_t == 0:
            self.nodes1d = np.array([(1 - 1 / np.sqrt(3)) / 2., (1 + 1 / np.sqrt(3)) / 2.])  # Order 4
            self.weights1d = np.array([1. / 2., 1. / 2.])
        
        elif deg_t == 1:
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

    # _temporal_ evolution form on the reference interval [0,1]
    def evolution_refEl(self, basis_id1, basis_id2):

        nq = self.nodes1d.shape[0]
        val = 0
        for k in range(0, nq):
            basis1 = legendre.basis1d(basis_id1, self.nodes1d[k])
            deriv_basis2 = legendre.basis1d_deriv(basis_id2, self.nodes1d[k])
            val += self.weights1d[k] * basis1 * deriv_basis2

        return val

    # Quadrature formula for \int_\hat{I_t} (f(t)*basis_t) d\hat{t}
    def projectionL2(self, ndof_x, f, a, b):

        p = self.deg_t
        nq = self.nodes1d.shape[0]
        l2NormFactor = 1. / np.sqrt((b - a))

        f_nodes = []
        t_tilde = np.zeros(nq)
        for k in range(0, nq):
            t_tilde[k] = functions.affineMap(a, b, self.nodes1d[k])
            f_nodes.append(f(t_tilde[k]).get_local())

        val = np.zeros((p + 1) * ndof_x)
        for k in range(0, p + 1):
            for q in range(0, nq):
                basisVal_t = legendre.basis1d(k, self.nodes1d[q])
                val[k * ndof_x: (k + 1) * ndof_x] += self.weights1d[q] * f_nodes[q] * basisVal_t

        return val * (b - a) * l2NormFactor

# End of file
