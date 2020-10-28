#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
cur_dir = str(Path().absolute())  # get working directory

import mesh_generator.io_mesh as meshio

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams

font = {'family': 'Dejavu Sans',
        'weight': 'normal',
        'size': 22}
rc('font', **font)
rcParams['lines.markersize'] = 12
rcParams['markers.fillstyle'] = 'none'
plt.rcParams["figure.figsize"] = [12, 12]

meshes_dir = cur_dir + "/meshes/"  # meshes directory
output_dir = cur_dir + "/figures/"  # output directory


def save_plot_mesh2d(mesh, filename):
    x, y = mesh.coordinates()[:, 0], mesh.coordinates()[:, 1]
    plt.triplot(x, y, triangles=mesh.cells())

    fig = plt.gcf()
    fig.savefig(filename, format='pdf', dpi=1000)
    plt.show()


# Unit square domain
def plot_meshes_test11():
    qu_lx = [3]
    for k in range(0, len(qu_lx)):
        qu_mesh_file = meshes_dir + "square_quasiUniform/mesh_l%d.h5" % qu_lx[k]
        filename = output_dir + "unitSquare_uniformTriMesh_l%d.pdf" % qu_lx[k]
        print(qu_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(qu_mesh_file)
        save_plot_mesh2d(mesh, filename)


# Gamma-shaped domain
def plot_meshes_test12():
    qu_lx = [2, 3]
    for k in range(0, len(qu_lx)):
        qu_mesh_file = meshes_dir + "gammaShaped_quasiUniform/mesh_l%d.h5" % qu_lx[k]
        filename = output_dir + "gammaShapedDomain_quasiUniform_l%d.pdf" % qu_lx[k]
        print(qu_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(qu_mesh_file)
        save_plot_mesh2d(mesh, filename)

    # deg 1
    bi_lx = [2, 3]
    for k in range(0, len(bi_lx)):
        bi_mesh_file = meshes_dir + "gammaShaped_bisecRefine/linear/mesh_l%d.h5" % bi_lx[k]
        filename = output_dir + "gammaShapedDomain_bisecRefine_deg1_l%d.pdf" % bi_lx[k]
        print(bi_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(bi_mesh_file)
        save_plot_mesh2d(mesh, filename)

    # deg 2
    bi_lx = [2, 3]
    for k in range(0, len(bi_lx)):
        bi_mesh_file = meshes_dir + "gammaShaped_bisecRefine/quadratic/mesh_l%d.h5" % bi_lx[k]
        filename = output_dir + "gammaShapedDomain_bisecRefine_deg2_l%d.pdf" % bi_lx[k]
        print(bi_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(bi_mesh_file)
        save_plot_mesh2d(mesh, filename)

    # deg 3
    bi_lx = [2, 3]
    for k in range(0, len(bi_lx)):
        bi_mesh_file = meshes_dir + "gammaShaped_bisecRefine/cubic/mesh_l%d.h5" % bi_lx[k]
        filename = output_dir + "gammaShapedDomain_bisecRefine_deg3_l%d.pdf" % bi_lx[k]
        print(bi_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(bi_mesh_file)
        save_plot_mesh2d(mesh, filename)


# Square domain, consists of two rectangular sectors
def plot_meshes_test13():
    qu_lx = [4]
    for k in range(0, len(qu_lx)):
        qu_mesh_file = meshes_dir + "square_twoPiecewise/quasiUniform/mesh_l%d.h5" % qu_lx[k]
        filename = output_dir + "square_twoPiecewise_quasiUniform_l%d.pdf" % qu_lx[k]
        print(qu_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(qu_mesh_file)
        save_plot_mesh2d(mesh, filename)


# Square domain, consists of two rectangular sectors
def plot_meshes_test14():
    bi_lx = [1]
    for k in range(0, len(bi_lx)):
        bi_mesh_file = meshes_dir + "square_twoPiecewise/bisecRefine/quadratic/mesh_l%d.h5" % bi_lx[k]
        filename = output_dir + "square_twoPiecewise_bisecRefine_deg2_l%d.pdf" % bi_lx[k]
        print(bi_mesh_file, "\n", filename)
        mesh = meshio.read_mesh_h5(bi_mesh_file)
        save_plot_mesh2d(mesh, filename)


if __name__ == "__main__":
    # plot_meshes_test11()
    # plot_meshes_test12()
    # plot_meshes_test13()
    plot_meshes_test14()

# END OF FILE
