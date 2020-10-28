#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib import rcParams

font = {'family': 'Dejavu Sans',
        'weight': 'normal',
        'size': 30}
rc('font', **font)
rcParams['lines.linewidth'] = 4
rcParams['lines.markersize'] = 18
rcParams['markers.fillstyle'] = 'none'


def plot_signal():
    file = open("../output/scattering_signal.txt", "r")
    N = int(file.readline())
    time_series = file.readline().split(',')
    raw_data = file.readline().split(',')

    time = np.zeros(N)
    pressure = np.zeros(N)
    for k in range(0, N):
        time[k] = float(time_series[k])
        pressure[k] = float(raw_data[k])

    file.close()

    signal = np.zeros(N)
    for k in range(1, N):
        h = time[k] + time[k - 1]
        signal[k] = signal[k - 1] + 0.5 * (pressure[k] + pressure[k - 1]) / h

    plt.plot(time, signal, 'b-')

    plt.xlabel('time $t$')
    plt.ylabel('$u_C (t)$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylim(-2E-3, 2E-3)

    fig = plt.gcf()
    fig.set_size_inches(18, 14)
    # plt.show()
    fig.savefig('output/scattering_signal.eps', format='eps', dpi=1000)


if __name__ == "__main__":
    plot_signal()

# End of file
