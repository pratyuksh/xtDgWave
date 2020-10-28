#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


# CLASS POLYGON #

class Polygon:

    def __init__(self):

        self.corners = None
        self.singular_corners = None
        self.refine_weights = None
        self.refine_flags = None
        self.tol = None

    # determines if a given point is a singular corner
    def is_a_singular_corner(self, x):

        num_singular_corners = self.singular_corners.shape[0]
        if num_singular_corners == 0:
            return False, 0

        for i in range(0, num_singular_corners):
            j = self.singular_corners[i]
            v = self.corners[j - 1]
            if la.norm(v - x) <= self.tol * la.norm(v):
                return True, i + 1

        return False, 0

    # sets the singular corners
    def set_singular_corners(self, singular_corners):
        self.singular_corners = singular_corners

    # sets local refinement weights for singular corners
    def set_refine_weights(self, refine_weights):
        self.refine_weights = refine_weights

    # sets local refinement flags for singular corners
    # used in bisection mesh refinement
    def set_refine_flags(self, refine_flags):
        self.refine_flags = refine_flags


##
class SquareDomain(Polygon):

    def __init__(self):
        self.tol = 1e-15

        self.left = 0.
        self.right = 1.
        self.bottom = 0.
        self.top = 1.

        self.add_corners()

        # default singular corners and refinement weights
        self.add_singular_corners()
        self.refine_weights = np.array([0.5] * self.singular_corners.shape[0])

    def add_corners(self):
        self.corners = np.array([[self.left, self.bottom],
                                 [self.right, self.bottom],
                                 [self.right, self.top],
                                 [self.left, self.top]])

    # indexing starts from 1
    def add_singular_corners(self):
        self.singular_corners = np.array([1])


##
class AngularDomain(Polygon):

    def __init__(self, angle):

        self.tol = 1e-15

        a = 0.5
        self.left = -a
        self.right = +a
        self.bottom = -a
        self.top = +a

        self.angle = angle
        self.add_corners()

        # default singular corners and refinement weights
        self.add_singular_corners()
        self.refine_weights = np.array([0.5] * self.singular_corners.shape[0])

    " Extract corners of polygonal domain out of given information "

    def add_corners(self):

        x_mid = .5 * (self.left + self.right)
        y_mid = .5 * (self.bottom + self.top)

        if 3 * np.pi / 2 + np.pi / 4 >= self.angle >= 3 * np.pi / 2 - np.pi / 4:
            " CASE: Vertex on the bottom: 5/4 Pi <= angle <= 7/4 Pi "
            self.corners = np.array([[self.left, self.bottom],
                                     [x_mid + np.tan(self.angle - 1.5 * np.pi), self.bottom],
                                     [x_mid, y_mid],
                                     [self.right, y_mid],
                                     [self.right, self.top],
                                     [self.left, self.top]])

        elif 3 * np.pi / 2 + np.pi / 4 < self.angle < 2 * np.pi:
            " CASE: Vertex on the rhs: 7/4 Pi <= angle < 2 Pi "
            self.corners = np.array([[self.left, self.bottom],
                                     [self.right, self.bottom],
                                     [self.right, y_mid - np.tan(2.0 * np.pi - np.tan(self.angle) * .5)],
                                     [x_mid, y_mid],
                                     [self.right, y_mid],
                                     [self.right, self.top],
                                     [self.left, self.top]])

        elif self.angle == np.pi:
            " CASE: angle =  np.pi "
            self.corners = np.array([[self.left, self.bottom],
                                     [self.right, self.bottom],
                                     [self.right, self.top],
                                     [self.left, self.top]])

        else:
            raise RuntimeError('ERROR in getUnifMesh: invalid angle=', self.angle / np.pi, 'pi.')

    # indexing starts from 1
    def add_singular_corners(self):
        self.singular_corners = np.array([3])


##
class SquareTwoPiecewiseDomain(Polygon):

    def __init__(self):
        self.tol = 1e-15
        self.add_corners()

        # default singular corners and refinement weights
        self.add_singular_corners()
        self.refine_weights = np.array([0.5] * self.singular_corners.shape[0])

    def add_corners(self):
        self.corners = np.array([[0, 0],
                                 [1.2, 0],
                                 [2, 0],
                                 [0, 1],
                                 [1.2, 1],
                                 [2, 1],
                                 [0, 2],
                                 [1.2, 2],
                                 [2, 2]])

    # indexing starts from 1
    def add_singular_corners(self):
        self.singular_corners = np.array([5])

# END OF FILE
