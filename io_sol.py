#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dolfin as df
import matplotlib.pyplot as plt


# plots the solution 1d
def plot_sol1d(u):
  
    p = df.plot(u)
    plt.show()


# plots the solution 2d
def plot_sol2d(u):
  
    p = df.plot(u)
    plt.colorbar(p)
    plt.show()


# writes the solution to a hdf5 file
def write_sol_h5(u, t, sol_name, sol_file):
	
	f = df.HDF5File(df.mpi_comm_world(), sol_file, 'w')
	f.write(u, sol_name, t)
	f.close()

	
# reads the solution from a hdf5 file
def read_sol_h5(sol_name, sol_file, u):
	
	f = df.HDF5File(df.mpi_comm_world(), sol_file, 'r')
	f.read(u, sol_name)
	f.close()
	

# writes the solution to a paraview format file
def write_sol_pvd(u, t, sol_file):
	
	#f = df.File(sol_file, "compressed")
        f = df.File(sol_file)
        f << u, t

	
# converts the solution from hdf5 to paraview format
def convert_sol_h5_to_vtk(p, t, sol_name, mesh_file, h5_sol_file, pvd_sol_file):
	
	mesh = df.Mesh(mesh_file)
	V = df.FunctionSpace(mesh, 'CG', p)
	u = df.Function(V)
	
	read_sol_h5(sol_name, h5_sol_file, u)
	write_sol_pvd(u, t, pvd_sol_file)


# write full-grid solution to paraview format
def write_solFG_pvd(cfg, dir_sol, t, u, lx, lt):
    pvd_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d.pvd"%lt
    print(' Write solution to pvd file: ', pvd_sol_file)
    write_sol_pvd(u, t, pvd_sol_file)


# write full-grid solution to hdf5 file
def write_solFG(cfg, dir_sol, tMesh, u, lx, lt):
    
    h5_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d.h5"%lt
    print(' Write solution to hdf5 file: ', h5_sol_file)
    
    if cfg['save xt sol']:
        write_xtdG_sol_time_series_h5(u, cfg['deg_t'], tMesh, cfg['system'], h5_sol_file)        
    else:
        write_sol_h5(u, tMesh[-1], cfg['system'], h5_sol_file)


# write full-grid reference solution to hdf5 file
def write_solFG_ref(cfg, dir_sol, tMesh, u, lx, lt):
    
    h5_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d_ref.h5"%lt
    print(' Write solution to file: ', h5_sol_file)
    write_sol_h5(u, tMesh[-1], cfg['system'], h5_sol_file)


# read full-grid solution from hdf5 file
def read_solFG(cfg, dir_sol, u, lx, lt):
    
    h5_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d.h5"%lt
    print(' Read solution from file: ', h5_sol_file)
    
    if cfg['save xt sol']:
        pass
        #read_xtdG_sol_time_series_h5(u, cfg['deg_t'], cfg['system'], h5_sol_file)
    else:
        read_sol_h5(cfg['system'], h5_sol_file, u)


# read full-grid reference solution from hdf5 file
def read_solFG_ref(cfg, dir_sol, u, lx, lt):
    
    h5_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d_ref.h5"%lt
    print(' Read solution from file: ', h5_sol_file)
    read_sol_h5(cfg['system'], h5_sol_file, u)


# write sparse-grid solution to hdf5 file
def write_solSG(cfg, dir_sol, tMeshes, u_solutions, L, L0, levels):
    
    nSG = len(uSolutions)
    Ldiff = L-L0
    
    for k in range(0, nSG):
        
        lx = levels[k,0]+1
        lt = levels[k,1]+1
        
        h5_sol_file = dir_sol+cfg['dump sol subdir']+cfg['test case']+"_lx%d"%lx+"_lt%d.h5"%lt
        print(' Write solution to file: ', h5_sol_file)
        
        if cfg['save xt sol']:
            write_xtdG_sol_time_series_h5(u_solutions[k], cfg['deg_t'], tMeshes[k%(Ldiff+1)], cfg['system'], h5_sol_file)        
        else:
            write_sol_h5(u_solutions[k], tMeshes[0][-1], cfg['system'], h5_sol_file)
    
	
# END OF FILE
