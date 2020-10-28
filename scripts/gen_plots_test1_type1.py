#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import utilities as utl

from matplotlib import rc
from matplotlib import rcParams

font = {'family': 'Dejavu Sans',
        'weight': 'normal',
        'size': 30}
rc('font', **font)
rcParams['lines.linewidth'] = 4
rcParams['lines.markersize'] = 18
rcParams['markers.fillstyle'] = 'none'


def plotData(data_x, data_y, ref_data_y, myPlotDict):
    fig, ax = plt.subplots()
    for k in range(0, len(data_y)):
        N = data_y[k].shape[0]
        ax.loglog(data_x[k][0:N], data_y[k], myPlotDict['data_markers'][k], label=myPlotDict['data_labels'][k])

    for k in range(0, len(ref_data_y)):
        N = ref_data_y[k].shape[0]
        ax.loglog(data_x[k][0:N], ref_data_y[k], myPlotDict['ref_data_markers'][k],
                  label=myPlotDict['ref_data_labels'][k])

    ax.legend(loc=myPlotDict['legend_loc'], shadow=False)
    plt.xlabel(myPlotDict['xlabel'])
    plt.ylabel(myPlotDict['ylabel'])
    plt.xlim(myPlotDict['xlim'])
    plt.ylim(myPlotDict['ylim'])
    plt.grid()

    fig = plt.gcf()
    fig.set_size_inches(18, 14)
    if myPlotDict['showPlot'] and not myPlotDict['savePlot']:
        plt.show()
    elif myPlotDict['savePlot']:
        fig.savefig(myPlotDict['outFileName'], format='eps', dpi=1000)


def FGvsSG_xErrL2L2_test1_smooth_type1(dir_fig, stabParamsType, showPlot, savePlot, plotId):
    ndim = 2

    deg = []
    inv_h = []
    ndof = []

    errL2 = []
    ref_errL2 = []

    errL2_v = []
    ref_errL2_v = []

    errL2_sigma = []
    ref_errL2_sigma = []

    # SG; L0t = 0, T = 1
    # deg_x = deg_t = 1; L0x = 0
    f1 = '../output/waveO1/type1/stabParams' + str(
        stabParamsType) + '/runSG_test1_quasiUniform_xErrL2L2_degx1_degt1.txt'
    deg_, _, inv_h_, ndof_, errL2_ = utl.read_outputSG(f1)

    deg.append(deg_)
    inv_h.append(inv_h_)
    ndof.append(ndof_)
    errL2_v.append(errL2_[:, 0])
    errL2_sigma.append(errL2_[:, 1])
    errL2.append((errL2_[:, 0] + errL2_[:, 1]))

    ref_const = 4e+2
    ref_errL2.append(ref_const / ndof[-1] ** 1)

    ref_const = 4e+2
    ref_errL2_v.append(ref_const / ndof[-1] ** 1)

    ref_const = 2e+2
    ref_errL2_sigma.append(ref_const / ndof[-1] ** 1)

    # deg_x = deg_t = 2; L0x = 0
    f2 = '../output/waveO1/type1/stabParams' + str(
        stabParamsType) + '/runSG_test1_quasiUniform_xErrL2L2_degx2_degt2.txt'
    deg_, _, inv_h_, ndof_, errL2_ = utl.read_outputSG(f2)

    deg.append(deg_)
    inv_h.append(inv_h_)
    ndof.append(ndof_)
    errL2_v.append(errL2_[:, 0])
    errL2_sigma.append(errL2_[:, 1])
    errL2.append((errL2_[:, 0] + errL2_[:, 1]))

    ref_const = 2e+4
    ref_errL2.append(ref_const / ndof[-1] ** 1.5)

    ref_const = 2e+4
    ref_errL2_v.append(ref_const / ndof[-1] ** 1.5)

    ref_const = 6e+3
    ref_errL2_sigma.append(ref_const / ndof[-1] ** 1.5)

    # FG; T = 1
    # deg_x = deg_t = 1
    f3 = '../output/waveO1/type1/stabParams' + str(
        stabParamsType) + '/runFG_test1_quasiUniform_xErrL2L2_degx1_degt1.txt'
    deg_, _, inv_h_, _, ndof_, errL2_ = utl.read_outputFG(f3)

    deg.append(deg_)
    inv_h.append(inv_h_)
    ndof.append(ndof_)
    errL2_v.append(errL2_[:, 0])
    errL2_sigma.append(errL2_[:, 1])
    errL2.append((errL2_[:, 0] + errL2_[:, 1]))

    ref_const = 5e+1
    ref_errL2.append(ref_const / ndof[-1] ** 0.67)

    ref_const = 5e+1
    ref_errL2_v.append(ref_const / ndof[-1] ** 0.67)

    ref_const = 3e+1
    ref_errL2_sigma.append(ref_const / ndof[-1] ** 0.67)

    # deg_x = deg_t = 2
    f4 = '../output/waveO1/type1/stabParams' + str(
        stabParamsType) + '/runFG_test1_quasiUniform_xErrL2L2_degx2_degt2.txt'
    deg_, _, inv_h_, _, ndof_, errL2_ = utl.read_outputFG(f4)

    deg.append(deg_)
    inv_h.append(inv_h_)
    ndof.append(ndof_)
    errL2_v.append(errL2_[:, 0])
    errL2_sigma.append(errL2_[:, 1])
    errL2.append((errL2_[:, 0] + errL2_[:, 1]))

    ref_const = 3e+2
    ref_errL2.append(ref_const / ndof[-1] ** 1)

    ref_const = 3e+2
    ref_errL2_v.append(ref_const / ndof[-1] ** 1)

    ref_const = 1e+2
    ref_errL2_sigma.append(ref_const / ndof[-1] ** 1)

    myPlotDict = {'showPlot': showPlot, 'savePlot': savePlot, 'xlabel': r'Number of degrees of freedom, $M_L$ [log]',
                  'legend_loc': 'upper right', 'data_markers': ['r-o', 'b-s'], 'ref_data_markers': ['r--', 'b-.'],
                  'xlim': [1e+2, 1e+8]}

    # plot error vs ndof
    if plotId == 1:
        # errorL2 vs h; deg = 1
        myPlotDict['data_labels'] = ['$p=1$, SG', '$p=1$, FG']
        indices = [0, 2]
        ref_indices = [0, 2]
        data_x = [ndof[i] for i in indices]

        data_y = [errL2_v[i] for i in indices]
        ref_data_y = [ref_errL2_v[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1})$', '$O(M_L^{-0.67})$']
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - \hat{v}_h \right||_{L^2(\Omega\times\{T\})}$ [log]'
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg1_v.eps"
        myPlotDict['ylim'] = [5e-5, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)

        data_y = [errL2_sigma[i] for i in indices]
        ref_data_y = [ref_errL2_sigma[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1})$', '$O(M_L^{-0.67})$']
        myPlotDict[
            'ylabel'] = r'Rel. error $\left|| \mathbf{\sigma} - \hat{\mathbf{\sigma_h}} \right||' \
                        r'_{L^2(\Omega\times\{T\})^{%d}}$ [log]' % ndim
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg1_sigma.eps"
        myPlotDict['ylim'] = [5e-5, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)

        data_y = [errL2[i] for i in indices]
        ref_data_y = [ref_errL2[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1})$', '$O(M_L^{-0.67})$']
        myPlotDict['ylabel'] = r'Total error $\mathcal{E}$ [log]'
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg1.eps"
        myPlotDict['ylim'] = [5e-5, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)

    if plotId == 2:
        # errorL2 vs h; deg = 2
        myPlotDict['data_labels'] = ['$p=2$, SG', '$p=2$, FG']
        indices = [1, 3]
        ref_indices = [1, 3]
        data_x = [ndof[i] for i in indices]

        data_y = [errL2_v[i] for i in indices]
        ref_data_y = [ref_errL2_v[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1.5})$', '$O(M_L^{-1})$']
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - \hat{v}_h \right||_{L^2(\Omega\times\{T\})}$ [log]'
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg2_v.eps"
        myPlotDict['ylim'] = [1e-7, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)

        data_y = [errL2_sigma[i] for i in indices]
        ref_data_y = [ref_errL2_sigma[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1.5})$', '$O(M_L^{-1})$']
        myPlotDict[
            'ylabel'] = r'Rel. error $\left|| \mathbf{\sigma} - \hat{\mathbf{\sigma_h}} \right||' \
                        r'_{L^2(\Omega\times\{T\})^{%d}}$ [log]' % ndim
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg2_sigma.eps"
        myPlotDict['ylim'] = [1e-7, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)

        data_y = [errL2[i] for i in indices]
        ref_data_y = [ref_errL2[i] for i in ref_indices]
        myPlotDict['ref_data_labels'] = ['$O(M_L^{-1.5})$', '$O(M_L^{-1})$']
        myPlotDict['ylabel'] = r'Total error $\mathcal{E}$ [log]'
        myPlotDict['outFileName'] = dir_fig + "FGvsSG_test1_smooth_type1_deg2.eps"
        myPlotDict['ylim'] = [1e-7, 2]
        plotData(data_x, data_y, ref_data_y, myPlotDict)


if __name__ == '__main__':
    dir_fig_ = "../figures/waveO1/"

    showPlot_ = True
    savePlot_ = False

    plotId_ = 2
    stabParamsType_ = 4
    FGvsSG_xErrL2L2_test1_smooth_type1(dir_fig_, stabParamsType_, showPlot_, savePlot_, plotId_)

# END OF FILE
