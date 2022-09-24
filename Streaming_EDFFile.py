#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pyedflib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def stackplot(marray, seconds=None, start_time=None, ylabels=None, ax=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels, ax=ax)
    plt.show()


def stackplot_t(tarray, seconds=None, start_time=None, ylabels=None, ax=None):
    """
    will plot a stack of traces one above the other assuming
    tarray.shape =  numSamples, numRows
    """
    data = tarray
    numSamples, numRows = tarray.shape
# data = np.random.randn(numSamples,numRows) # test data
# data.shape = numSamples, numRows
    if seconds:
        t = seconds * np.arange(numSamples, dtype=float)/numSamples
# import pdb
# pdb.set_trace()
        if start_time:
            t = t+start_time
            xlm = (start_time, start_time+seconds)
        else:
            xlm = (0,seconds)

    else:
        t = np.arange(numSamples, dtype=float)
        xlm = (0,numSamples)

    ticklocs = []
    if ax is None:
        ax = plt.subplot(111)
    plt.xlim(*xlm)
    # xticks(np.linspace(xlm, 10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin)*0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax
    plt.ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:,np.newaxis], data[:,i,np.newaxis])))
        # print "segs[-1].shape:", segs[-1].shape
        ticklocs.append(i*dr)

    offsets = np.zeros((numRows,2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None,
                           )

    ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
    # if not plt.ylabels:
    plt.ylabels = ["%d" % ii for ii in range(numRows)]
    ax.set_yticklabels(ylabels)

    plt.xlabel('time (s)')


def test_stacklineplot():
    numSamples, numRows = 800, 5
    data = np.random.randn(numRows, numSamples)  # test data
    stackplot(data, 10.0)

def animate(frame):
    global offset
    for i in np.arange(n):
        sigbufs_plot[i,:] = sigbufs[i][offset:n_plot + offset]
    ax1.clear()
    stackplot_t(np.transpose(sigbufs_plot[:, :n_plot]), ylabels=signal_labels, ax=ax1)
    offset += dt

if __name__ == '__main__':
    #f = pyedflib.data.test_generator()
    file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/01_tcp_ar/003/00000355/s003_2013_01_04/00000355_s003_t000.edf"
    #file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/01_tcp_ar/018/00001819/s002_2012_04_30/00001819_s002_t000.edf"

    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/01_tcp_ar/027/00002744/s002_2011_04_13/00002744_s002_t001.edf"
    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/01_tcp_ar/066/00006687/s001_2010_06_02/00006687_s001_t000.edf"
    f = pyedflib.EdfReader(file)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    n_min = f.getNSamples()[0]
    sigbufs = [np.zeros(f.getNSamples()[i]) for i in np.arange(n)]
    for i in np.arange(n):
        sigbufs[i] = f.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])
            
    duration = f.getFileDuration()
    f._close()
    del f
    dt = int(duration/5)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)    
    n_plot = np.min((n_min, 100))
    sigbufs_plot = np.zeros((n, n_plot))    
    offset = 0
    ani = animation.FuncAnimation(fig, animate, interval=200)
    plt.show()