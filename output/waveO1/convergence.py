#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utilities as utl

from decimal import Decimal

def convgRateWrtH (data_x, data_y):
    N = data_x.size
    #z = np.polyfit(np.log2(data_x), np.log2(data_y), 1)
    #z = np.polyfit(np.log2(data_x[1:N]), np.log2(data_y[1:N]), 1)
    z = np.polyfit(np.log2(data_x[N-3:N]), np.log2(data_y[N-3:N]), 1)
    return round(z[0],2)


def FG_xErrL2L2_test1_smooth(ptype, stabParams, deg):

    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_smooth_quasiUniform_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, _, ndof, errL2 = utl.read_outputFG(f1)
    
    for k in range(errL2.shape[0]-1,errL2.shape[0]):
        print(k, "%.4E"%Decimal(errL2[k,0]), "%.4E"%Decimal(errL2[k,1]))
    print("\n\n")
    for k in range(errL2.shape[0]-3,errL2.shape[0]-1):
        val1 = np.log2(errL2[k,0]) - np.log2(errL2[k+1,0])
        val2 = np.log2(errL2[k,1]) - np.log2(errL2[k+1,1])
        print(k, "%4.2f"%val1, "%4.2f"%val2)
    
    rate_v = convgRateWrtH (invH, errL2[:,0])
    rate_sigma = convgRateWrtH (invH, errL2[:,1])
    print('\nConvergence rate of v: ', rate_v)
    print('Convergence rate of sigma: ', rate_sigma)


def FG_xErrL2L2_test1_singular(ptype, stabParams, deg):

    #f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_singular_quasiUniform_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_singular_bisectionRefined_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, _, ndof, errL2 = utl.read_outputFG(f1)
    
    for k in range(errL2.shape[0]-1,errL2.shape[0]):
        print(k, "%.4E"%Decimal(errL2[k,0]), "%.4E"%Decimal(errL2[k,1]))
    print("\n\n")
    for k in range(errL2.shape[0]-3,errL2.shape[0]-1):
        val1 = np.log2(errL2[k,0]) - np.log2(errL2[k+1,0])
        val2 = np.log2(errL2[k,1]) - np.log2(errL2[k+1,1])
        print(k, "%4.2f"%val1, "%4.2f"%val2)
    
    rate_v = convgRateWrtH (invH, errL2[:,0])
    rate_sigma = convgRateWrtH (invH, errL2[:,1])
    print('\nConvergence rate of v: ', rate_v)
    print('Convergence rate of sigma: ', rate_sigma)


def SG_xErrL2L2_test1_smooth(ptype, stabParams, deg):

    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runSG_test1_smooth_quasiUniform_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, ndof, errL2 = utl.read_outputSG(f1)
    
    rate_v = convgRateWrtH (ndof, errL2[:,0])
    rate_sigma = convgRateWrtH (ndof, errL2[:,1])
    print('\nConvergence rate of v: ', rate_v)
    print('Convergence rate of sigma: ', rate_sigma)


def SG_xErrL2L2_test1_singular(ptype, stabParams, deg):

    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runSG_test1_singular_bisectionRefined_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, ndof, errL2 = utl.read_outputSG(f1)
    
    rate_v = convgRateWrtH (ndof, errL2[:,0])
    rate_sigma = convgRateWrtH (ndof, errL2[:,1])
    print('\nConvergence rate of v: ', rate_v)
    print('Convergence rate of sigma: ', rate_sigma)
    

def FG_xtErrDG_test1_smooth(ptype, stabParams, deg):

    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_smooth_quasiUniform_xtErrDG_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, _, ndof, errDG_ = utl.read_outputFG(f1)
    
    errDG = np.sqrt(errDG_[:,0]**2 + errDG_[:,1]**2)
    
    for k in range(errDG.size-1,errDG.size):
        print(k, "%.4E"%Decimal(errDG[k]))
    print("\n\n")
    for k in range(errDG.size-3,errDG.size-1):
        val = np.log2(errDG[k]) - np.log2(errDG[k+1])
        print(k, "%4.2f"%val)
    
    rate = convgRateWrtH (invH, errDG)
    print('\nConvergence rate of DG error: ', rate)

def FG_xtErrDG_test1_singular(ptype, stabParams, deg):

    #f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_singular_quasiUniform_xtErrDG_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    f1 = 'type'+str(ptype)+'/stabParams'+str(stabParams)+'/runFG_test1_singular_bisectionRefined_xErrL2L2_degx'+str(deg)+'_degt'+str(deg)+'.txt'
    deg, _, invH, _, ndof, errL2 = utl.read_outputFG(f1)
    deg, _, invH, _, ndof, errDG_ = utl.read_outputFG(f1)
    
    errDG = np.sqrt(errDG_[:,0]**2 + errDG_[:,1]**2)
    
    for k in range(errDG.size-1,errDG.size):
        print(k, "%.4E"%Decimal(errDG[k]))
    print("\n\n")
    for k in range(errDG.size-3,errDG.size-1):
        val = np.log2(errDG[k]) - np.log2(errDG[k+1])
        print(k, "%4.2f"%val)
    
    rate = convgRateWrtH (invH, errDG)
    print('\nConvergence rate of DG error: ', rate)


if __name__=='__main__':
    """
    print("\n\nFull-grid smooth solution:\n")
    ptype = 1
    deg = 3
    for stabParams in range(1,5):
        FG_xErrL2L2_test1_smooth(ptype, stabParams, deg)
    """
    """
    print("\n\nFull-grid singular solution:\n")
    ptype = 1
    deg = 3
    #FG_xErrL2L2_test1_singular(ptype, 1, deg)
    for stabParams in range(1,5):
        FG_xErrL2L2_test1_singular(ptype, stabParams, deg)
    """
    """
    print("\n\nSparse-grid smooth solution:\n")
    ptype = 1
    deg = 2
    for stabParams in range(1,6):
        SG_xErrL2L2_test1_smooth(ptype, stabParams, deg)

    print("\n\nSparse-grid singular solution:\n")
    ptype = 1
    deg = 2
    for stabParams in range(1,6):
        SG_xErrL2L2_test1_singular(ptype, stabParams, deg)
    """
    
    """
    print("\n\nFull-grid smooth solution:\n")
    ptype = 2
    deg = 3
    for stabParams in range(1,5):
        FG_xtErrDG_test1_smooth(ptype, stabParams, deg)
    """
    
    print("\n\nFull-grid singular solution:\n")
    ptype = 2
    deg = 3
    # FG_xtErrDG_test1_singular(ptype, 1, deg)
    for stabParams in range(1,5):
        FG_xtErrDG_test1_singular(ptype, stabParams, deg)


## END OF FILE
