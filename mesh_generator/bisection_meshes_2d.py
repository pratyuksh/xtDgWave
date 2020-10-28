#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import fenics
import mshr


# CLASS: Bisection Mesh Refinement
class BisectionRefinement:

    def __init__(self, polygon):

        self.polygon = polygon
        # self.convert_to_fenics_format()

        self.factor = 0.49  # must be < 0.5
        if polygon:
            self.R0 = self.calc_R0()

    # converts the geometry corner c_k to fenics.Point(c_k)
    def convert_to_fenics_format(self, z):

        x = z[0]
        y = z[1]
        return fenics.Point(x, y)

    # calculates R0 for corners
    def calc_R0(self):

        num_corners = len(self.polygon.corners)
        R0 = np.zeros(num_corners)
        for j in range(0, num_corners):
            d_min = 1e+10
            for i in range(0, num_corners):
                if i != j:
                    d = la.norm(self.polygon.corners[i] - self.polygon.corners[j])
                    d_min = min(d_min, d)
            R0[j] = d_min * self.factor

        return R0

    # uniform refinement
    def uniform(self, h0, mesh):

        while mesh.hmax() > h0:
            cell_markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim())
            cell_markers.set_all(True)
            mesh = fenics.refine(mesh, cell_markers)

        return mesh

    # local refinement
    def local(self, deg, h0, mesh):

        # TOL = fenics.DOLFIN_EPS_LARGE
        singular_corners = self.polygon.corners[self.polygon.singular_corners - 1, :]
        num_sin_corners = len(singular_corners)

        for i in range(num_sin_corners):
            if self.polygon.refine_flags[i] == 1:

                R0 = self.R0[self.polygon.singular_corners[i] - 1]
                corner = self.convert_to_fenics_format(singular_corners[i, :])

                gamma = 1. - self.polygon.refine_weights[i]
                K0 = -np.log2(h0) * (deg + 1.) / gamma - 1
                K = np.ceil(K0)
                NLocRef = int(2 * K + 1)
                print(h0, gamma, K, NLocRef, R0)

                for j in range(NLocRef):
                    weight = 1. - gamma
                    expo = -j * (deg + weight) / (2. * (deg + 1.))
                    h_min = h0 * np.power(2., expo)
                    R_max = R0 * np.power(2., -j / 2.)
                    cell_markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim())
                    cell_markers.set_all(False)
                    count = 0
                    for cell in fenics.cells(mesh):
                        h = max(e.length() for e in fenics.edges(cell))
                        # h=2*cell.circumradius()
                        dist = fenics.Cell.distance(cell, corner)
                        if dist < R_max:
                            if h > h_min:
                                cell_markers[cell] = True
                                count += 1
                    mesh = fenics.refine(mesh, cell_markers)

        return mesh


# CLASS: Bisection Mesh Generator 2d, triangulation
class BisectionMeshGenerator2d:

    def __init__(self, polygon):
        self.polygon = polygon
        self.bis_refine = BisectionRefinement(polygon)

        self.corners = None
        self.convert_to_fenics_format()

    # converts the geometry corner c_k to fenics.Point(c_k)
    def convert_to_fenics_format(self):
        num_corners = self.polygon.corners.shape[0]
        self.corners = []
        for k in range(0, num_corners):
            x = self.polygon.corners[k, 0]
            y = self.polygon.corners[k, 1]
            self.corners.append(fenics.Point(x, y))

    # generates quasi-uniform mesh
    def uniform(self, h0):
        domain = mshr.Polygon(self.corners)
        mesh = mshr.generate_mesh(domain, h0)
        return mesh

    # generates bisection refined mesh
    def bisection(self, deg, h0_init, h0):
        mesh = self.bisection_uniform(h0_init, h0)  # Step 1: uniform bisection refinement
        mesh = self.bisection_local(deg, mesh, h0)  # Step 2: local bisection refinement
        return mesh

    # generates _uniform_ bisection refined mesh
    def bisection_uniform(self, h0_init, h0):
        mesh_init = self.uniform(h0_init)
        mesh = self.bis_refine.uniform(h0, mesh_init)
        return mesh

    # generates _locally_ bisection refined mesh
    def bisection_local(self, deg, mesh, h0):
        mesh = self.bis_refine.local(deg, h0, mesh)
        return mesh

    # writes the Mesh to an output file
    def write(mesh, mesh_file):
        fenics.File(mesh_file) << mesh

# END OF FILE
