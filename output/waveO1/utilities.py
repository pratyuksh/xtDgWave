#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from matplotlib import rc
from matplotlib import rcParams
font = {'family' : 'Dejavu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)
rcParams['lines.markersize'] = 12
rcParams['markers.fillstyle'] = 'none'


def print_config(cfg):
    for key in cfg:
        print ("\t", key,":",cfg[key])


def write_outputFG(filename, cfg, inv_hx, inv_ht, ndof, errorBuf):
    
    file = open(filename,"w")
    
    if cfg['save xt sol']:
        errorType = 'xtErr_'+cfg['error type']
    else:
        errorType = 'xErr_'+cfg['error type']
    
    data = errorType+', '+str(cfg['deg_x_v'])+', '+str(cfg['deg_t'])+'\n'
    file.write(data)
    print(errorBuf)
    N = inv_hx.shape[0]
    for k in range(0,N):
        data = str(inv_hx[k])+', '+str(inv_ht[k])+', '+str(ndof[k])\
               +', '+str(errorBuf[k,0])+', '+str(errorBuf[k,1])+'\n'
        file.write(data)
    file.close()


def read_outputFG(filename):
    
    file = open(filename,"r")
    
    data = file.readlines()
    N = len(data)-1
    
    ## read deg_x, deg_t from first line    
    line = data[0]
    line_ = line.split(', ')
    deg_x = int(line_[1])
    deg_t = int(line_[2])
        
    inv_hx = np.zeros(N, dtype=np.int32)
    inv_ht = np.zeros(N, dtype=np.int32)
    ndof = np.zeros(N, dtype=np.int32)
    errorBuf = np.zeros((N,2), dtype=np.float64)
    ## read data
    for k in range(1,len(data)):
        line = data[k]
        line_ = line.split(', ')
        
        inv_hx[k-1] = int(line_[0])
        inv_ht[k-1] = int(line_[1])
        ndof[k-1] = int(line_[2])
        errorBuf[k-1,0] = float(line_[3])
        errorBuf[k-1,1] = float(line_[4])
    
    file.close()
    
    return deg_x, deg_t, inv_hx, inv_ht, ndof, errorBuf


def write_outputSG(filename, cfg, inv_h, ndof, errorBuf):
    
    file = open(filename,"w")
    
    if cfg['save xt sol']:
        errorType = 'xtErr_'+cfg['error type']
    else:
        errorType = 'xErr_'+cfg['error type']
    
    data = errorType+', '+str(cfg['deg_x_v'])+', '+str(cfg['deg_t'])+'\n'
    file.write(data)
    
    N = inv_h.shape[0]
    for k in range(0,N):
        data = str(inv_h[k])+', '+str(ndof[k])\
               +', '+str(errorBuf[k,0])+', '+str(errorBuf[k,1])+'\n'
        file.write(data)
    file.close()


def read_outputSG(filename):
    
    file = open(filename,"r")
    
    data = file.readlines()
    N = len(data)-1
    
    ## read deg_x, deg_t from first line    
    line = data[0]
    line_ = line.split(', ')
    deg_x = int(line_[1])
    deg_t = int(line_[2])
        
    inv_h = np.zeros(N, dtype=np.int64)
    ndof = np.zeros(N, dtype=np.int64)
    errorBuf = np.zeros((N,2), dtype=np.float64)
    ## read data
    for k in range(1,len(data)):
        line = data[k]
        line_ = line.split(', ')
        
        inv_h[k-1] = int(line_[0])
        ndof[k-1] = int(line_[1])
        errorBuf[k-1,0] = float(line_[2])
        errorBuf[k-1,1] = float(line_[3])
    
    file.close()
    
    return deg_x, deg_t, inv_h, ndof, errorBuf


def plotL2ErrorVsNdof(d, p, ndof_FG, err_FG, ndof_SG, err_SG):
    
    m_FG = (p+1)/(d+1)
    m_SG = (p+1)/d
    ref_FG = 3e+1/ndof_FG**m_FG
    ref_SG = 5e+2/ndof_SG**m_SG
    
    fig, ax = plt.subplots()
    ax.loglog(ndof_FG, err_FG, 'k-x', label='FG')
    ax.loglog(ndof_SG, err_SG, 'r-o', label='SG')
    ax.loglog(ndof_FG, ref_FG, 'k--x', label='$O(M_h^{%.2f})$'%m_FG)
    ax.loglog(ndof_SG, ref_SG, 'r--o', label='$O(M_h^{%.2f})$'%m_SG)
    
    legend = ax.legend(loc='lower left', shadow=False)
    
    plt.xlabel(r'$M_h$ [log]')
    plt.ylabel(r'Rel. error in ${L^2(Q)}$ norm [log]')
    plt.show()


def plotL2ErrorVsMeshSize(d, p, mesh_type, inv_hx, err_L2, showPlot, savePlot, filename):
    
    ## choose reference parameters
    if (mesh_type == ""):
        m = p+1
        factor = 2e+0
        
    elif (mesh_type == "quasi-uniform"):
        m = 2./3.
        if d == 1:
            factor = 1e+0
        else:
            factor = 4e-2
    
    elif mesh_type == "graded":
        m = p+0.5
        if d == 1:
            factor = 1e+0
        else:
            factor = 2e-2
    
    elif mesh_type == "bisection-refined":
        m = p+1
        if d == 1:
            factor = 1e+0
        else:
            factor = 8e-2
    
    ref_L2 = factor/inv_hx**m
    
    fig, ax = plt.subplots()
    ax.loglog(inv_hx, err_L2, 'r-x', label='$L^2$-norm')
    ax.loglog(inv_hx, ref_L2, 'k--x', label='$r_{L^2}$ = %.2f'%m)
    
    legend = ax.legend(loc='lower left', shadow=False)
    
    plt.xlabel(r'$h_x^{-1}$ [log]')
    plt.ylabel(r'Rel. error in ${L^2(\Omega)}$ norm [log]')
    
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    
    if showPlot:
        plt.show()
    if savePlot:
        fig.savefig(filename, format='eps', dpi=1000)


# END OF FILE
