#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import lxml.etree as etree
import h5py

plt.rcParams["figure.figsize"] = [16, 10]


# plots a 2d mesh
def plot_mesh2d(mesh):
    x, y = mesh.coordinates()[:, 0], mesh.coordinates()[:, 1]
    plt.triplot(x, y, triangles=mesh.cells())
    plt.show()


# plots the mesh
def plot_mesh(mesh):
    df.plot(mesh)
    plt.show()


# plots the mesh file
def plot_mesh_file(mesh_file):
    mesh = read_mesh_h5(mesh_file)
    df.plot(mesh)
    plt.show()


# writes the mesh to an xml file
def write_mesh_xml(mesh, mesh_file):
    df.File(mesh_file) << mesh


# writes the mesh to an hdf5 file
def write_mesh_h5(mesh, mesh_file):
    outFile = df.HDF5File(mesh.mpi_comm(), mesh_file, 'w')
    outFile.write(mesh, 'mesh')
    outFile.close()


# reads the mesh from an xml file
def read_mesh_xml(mesh_file):
    mesh = df.Mesh(mesh_file)
    return mesh


# reads the mesh from an hdf5 file
def read_mesh_h5(mesh_file):
    use_partition_from_file = False
    mesh = df.Mesh(df.mpi_comm_self())

    inFile = df.HDF5File(mesh.mpi_comm(), mesh_file, 'r')
    inFile.read(mesh, 'mesh', use_partition_from_file)
    inFile.close()

    return mesh


# converts the mesh from xml to hdf5 format
def convert_mesh_xml_to_h5(xml_mesh_file, h5_mesh_file):
    mesh = df.Mesh(xml_mesh_file)
    plot_mesh(mesh)
    outFile = df.HDF5File(mesh.mpi_comm(), h5_mesh_file, 'w')
    outFile.write(mesh, 'mesh')
    outFile.close()


# converts the mesh from xml to paraview format
def convert_mesh_xml_to_pvd(xml_mesh_file, pvd_mesh_file):
    mesh = df.Mesh(xml_mesh_file)
    df.File(pvd_mesh_file) << mesh


# converts the mesh from hdf5 to paraview format
def convert_mesh_h5_to_pvd(h5_mesh_file, pvd_mesh_file):
    use_partition_from_file = False
    mesh = df.Mesh()

    inFile = df.HDF5File(mesh.mpi_comm(), h5_mesh_file, 'r')
    inFile.read(mesh, 'mesh', use_partition_from_file)
    inFile.close()

    df.File(pvd_mesh_file) << mesh


# converts the mesh from hdf5 to vtk format
def convert_mesh_h5_to_vtk(h5_mesh_file, vtk_mesh_file):
    use_partition_from_file = False
    mesh = df.Mesh()

    inFile = df.HDF5File(mesh.mpi_comm(), h5_mesh_file, 'r')
    inFile.read(mesh, 'mesh', use_partition_from_file)
    inFile.close()

    writeMesh_vtk(mesh.coordinates(), mesh.cells(), vtk_mesh_file)


## 

# reads vertices and cells from a .xml FEniCS file
def readMesh2d_xml(mesh_file):
    # plot_mesh_file(mesh_file)

    # create an element tree object
    tree = ET.parse(mesh_file)
    mesh = tree.getroot()
    vertices_obj = mesh[0][0]
    cells_obj = mesh[0][1]

    # get number of vertices and cells
    num_vertices = int(vertices_obj.get('size'))
    num_cells = int(cells_obj.get('size'))

    # read vertices
    vertices = []
    for k in range(0, num_vertices):
        x = float(vertices_obj[k].get('x'))
        y = float(vertices_obj[k].get('y'))
        vertices.append([x, y])

    # read cells
    cells = []
    for k in range(0, num_cells):
        v0 = int(cells_obj[k].get('v0'))
        v1 = int(cells_obj[k].get('v1'))
        v2 = int(cells_obj[k].get('v2'))
        cells.append([v0, v1, v2])

    return np.asarray(vertices), np.asarray(cells)


# reads vertices and cells from a raw .dat file generated from LNG_FEM
def readMesh2d_LngFem_raw(mesh_file_nodes, mesh_file_elements):
    # read vertices
    f = open(mesh_file_nodes, "r")
    vertices = []

    x = f.readline()
    y = f.readline()

    x = x.strip().split(' ')
    y = y.strip().split(' ')
    for k in range(0, len(x)):
        vertices.append([float(x[k]), float(y[k])])
    vertices = np.asarray(vertices)
    f.close()

    # read cells
    f = open(mesh_file_elements, "r")
    cells = []

    v0 = f.readline()
    v1 = f.readline()
    v2 = f.readline()

    v0 = v0.strip().split(' ')
    v1 = v1.strip().split(' ')
    v2 = v2.strip().split(' ')
    for k in range(0, len(v0)):
        cells.append([int(v0[k]) - 1, int(v1[k]) - 1, int(v2[k]) - 1])
    cells = np.asarray(cells)

    f.close()
    return np.asarray(vertices), np.asarray(cells)


# reads vertices and cells from a .dat file
def readMesh2d_dat(mesh_file):
    # open .dat file to read
    f = open(mesh_file, "r")

    # read mesh info
    line = f.readline().split(',')
    num_vertices = int(line[0])
    num_cells = int(line[1])

    # read vertex coordinates
    vertices = []
    for k in range(0, num_vertices):
        line = f.readline().split(',')
        x = float(line[0])
        y = float(line[1])
        vertices.append([x, y])

    # read cell connectivity
    cells = []
    for k in range(0, num_cells):
        line = f.readline().split(',')
        v0 = int(line[0])
        v1 = int(line[1])
        v2 = int(line[2])
        cells.append([v0, v1, v2])

    f.close()
    return np.asarray(vertices), np.asarray(cells)


# reads vertices and cells from a .dat file
def readMesh3d_dat(mesh_file):
    # open .dat file to read
    f = open(mesh_file, "r")

    # read mesh info
    line = f.readline().split(',')
    num_vertices = int(line[0])
    num_cells = int(line[1])

    # read vertices
    vertices = []
    for k in range(0, num_vertices):
        line = f.readline().split(',')
        x = float(line[0])
        y = float(line[1])
        z = float(line[2])
        vertices.append([x, y, z])

    # read cells
    cells = []
    for k in range(0, num_cells):
        line = f.readline().split(',')
        v0 = int(line[0])
        v1 = int(line[1])
        v2 = int(line[2])
        v3 = int(line[3])
        cells.append(np.sort([v0, v1, v2, v3]))

    f.close()
    return np.asarray(vertices), np.asarray(cells)


# writes 2d/3d mesh to a .hdf5 file
def writeMesh_h5(vertices, cells, h5_file):
    # num_vertices = vertices.shape[0]
    num_cells = cells.shape[0]

    hf = h5py.File(h5_file, 'w')
    mesh = hf.create_group('mesh')

    mesh.create_dataset('cell_indices', data=np.arange(0, num_cells))
    mesh.create_dataset('coordinates', data=vertices)
    topology = mesh.create_dataset('topology', data=cells)

    if cells.shape[1] == 3:
        topology.attrs.create('celltype', np.string_("triangle"))
    elif cells.shape[1] == 4:
        topology.attrs.create('celltype', np.string_("tetrahedron"))
    topology.attrs.create('partition', np.array([0], dtype=np.uint64))

    hf.close()


# writes a 2d/3d mesh to a .vtk file
def writeMesh_vtk(vertices, cells, vtk_file):
    ndim = vertices.shape[1]

    # open .vtk file to write
    f = open(vtk_file, "w")
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Generated by MFEM\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # write POINTS
    f.write("POINTS %d double\n" % (vertices.shape[0]))
    if ndim == 2:
        for k in range(0, vertices.shape[0]):
            f.write("%10.8f" % vertices[k, 0] + " %10.8f" % vertices[k, 1] + " 0\n")

    elif ndim == 3:
        for k in range(0, vertices.shape[0]):
            f.write("%10.8f" % vertices[k, 0] + " %10.8f" % vertices[k, 1] + " %10.8f\n" % vertices[k, 2])

    # write CELLS
    f.write("CELLS %d" % (cells.shape[0]) + " %d\n" % (cells.shape[0] * (1 + cells.shape[1])))
    if ndim == 2:
        for k in range(0, cells.shape[0]):
            f.write("%d" % cells.shape[1] + " %d" % cells[k, 0] + " %d" % cells[k, 1] + " %d\n" % cells[k, 2])

    elif ndim == 3:
        for k in range(0, cells.shape[0]):
            f.write("%d" % cells.shape[1] + " %d" % cells[k, 0] + " %d" % cells[k, 1] + " %d" % cells[k, 2] + " %d\n" %
                    cells[k, 3])

    # write CELL_TYPES
    f.write("CELL_TYPES %d\n" % (cells.shape[0]))
    cell_type = None
    if ndim == 2:
        cell_type = 5
    elif ndim == 3:
        cell_type = 10
    for k in range(0, cells.shape[0]):
        f.write("%d\n" % cell_type)

    # write CELL_DATA
    f.write("CELL_DATA %d\n" % (cells.shape[0]))
    f.write("SCALARS material int\nLOOKUP_TABLE default\n")
    for k in range(0, cells.shape[0]):
        f.write("1\n")

    f.close()


# writes a 2d/3d mesh to MFEM .mesh file
def writeMesh_mfem(vertices, cells, bdry_attribs, mesh_file):
    ndim = vertices.shape[1]

    # open .mesh file to write
    f = open(mesh_file, "w")
    f.write("MFEM mesh v1.0\n")
    f.write("\ndimension\n")
    f.write("%d\n" % ndim)

    # write elements
    f.write("\nelements\n")
    f.write("%d\n" % (cells.shape[0]))
    if ndim == 2:
        for k in range(0, cells.shape[0]):
            f.write("1 2 %d" % cells[k, 0] + " %d" % cells[k, 1] + " %d\n" % cells[k, 2])

    elif ndim == 3:
        for k in range(0, cells.shape[0]):
            f.write("1 4 %d" % cells[k, 0] + " %d" % cells[k, 1] + " %d" % cells[k, 2] + " %d\n" % cells[k, 3])

    # write boundary
    f.write("\nboundary\n")
    f.write("%d\n" % (bdry_attribs.shape[0]))
    if ndim == 2:
        for k in range(0, bdry_attribs.shape[0]):
            f.write("%d" % bdry_attribs[k, 0] + " 1 %d" % bdry_attribs[k, 1] + " %d\n" % bdry_attribs[k, 2])

    elif ndim == 3:
        for k in range(0, bdry_attribs.shape[0]):
            f.write("%d" % bdry_attribs[k, 0] + " 2 %d" % bdry_attribs[k, 1] + " %d" % bdry_attribs[k, 2] + " %d\n" %
                    bdry_attribs[k, 3])

    # write vertices
    f.write("\nvertices\n")
    f.write("%d\n" % (vertices.shape[0]))
    f.write("%d\n" % (vertices.shape[1]))
    if ndim == 2:
        for k in range(0, vertices.shape[0]):
            f.write("%.15f" % vertices[k, 0] + " %.15f\n" % vertices[k, 1])

    elif ndim == 3:
        for k in range(0, vertices.shape[0]):
            f.write("%.15f" % vertices[k, 0] + " %.15f" % vertices[k, 1] + " %.15f\n" % vertices[k, 2])

    f.close()


# writes 2d mesh to a .xml file
def writeMesh2d_xml(vertices, cells, xml_file):
    num_vertices = vertices.shape[0]
    num_cells = cells.shape[0]

    # create an element tree object
    dolfin = etree.Element("dolfin")
    mesh = etree.SubElement(dolfin, "mesh", celltype="triangle", dim="2")

    # write vertices
    nodes = etree.SubElement(mesh, "vertices", size=str(num_vertices))
    for k in range(0, num_vertices):
        etree.SubElement(nodes, "vertex", index=str(k), x=str(vertices[k, 0]), y=str(vertices[k, 1]))

    # write cells
    elements = etree.SubElement(mesh, "cells", size=str(num_cells))
    for k in range(0, num_cells):
        etree.SubElement(elements, "triangle", index=str(k), v0=str(cells[k, 0]), v1=str(cells[k, 1]),
                         v2=str(cells[k, 2]))

    tree = etree.ElementTree(dolfin)
    tree.write(xml_file, pretty_print=True)


# writes 3d mesh to a .xml file
def writeMesh3d_xml(vertices, cells, xml_file):
    num_vertices = vertices.shape[0]
    num_cells = cells.shape[0]

    # create an element tree object
    dolfin = etree.Element("dolfin")
    mesh = etree.SubElement(dolfin, "mesh", celltype="tetrahedron", dim="3")

    # write vertices
    nodes = etree.SubElement(mesh, "vertices", size=str(num_vertices))
    for k in range(0, num_vertices):
        etree.SubElement(nodes, "vertex", index=str(k), x=str(vertices[k, 0]), y=str(vertices[k, 1]),
                         z=str(vertices[k, 2]))

    # write cells
    elements = etree.SubElement(mesh, "cells", size=str(num_cells))
    for k in range(0, num_cells):
        etree.SubElement(elements, "triangle", index=str(k), v0=str(cells[k, 0]), v1=str(cells[k, 1]),
                         v2=str(cells[k, 2]), v3=str(cells[k, 3]))

    tree = etree.ElementTree(dolfin)
    tree.write(xml_file, pretty_print=True)


# writes vertices, cells and ext_facets to a .dat file
def writeMesh2d_dat(vertices, cells, ext_facets, dat_file):
    # open .dat file to write
    f = open(dat_file, "w")
    f.write("%d" % (vertices.shape[0]) + ",%d" % (cells.shape[0]) + ",%d\n" % (ext_facets.shape[0]))

    # write vertices
    for k in range(0, vertices.shape[0]):
        f.write("%10.8f" % vertices[k, 0] + ",%10.8f\n" % vertices[k, 1])

    # write cells
    for k in range(0, cells.shape[0]):
        f.write("%d" % cells[k, 0] + ",%d" % cells[k, 1] + ",%d\n" % cells[k, 2])

    # write boundary facets
    for k in range(0, ext_facets.shape[0]):
        f.write("%d" % ext_facets[k, 0] + ",%d\n" % ext_facets[k, 1])

    f.close()

# END OF FILE
