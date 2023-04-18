from __future__ import division, print_function, absolute_import

import numpy as np
import time
import glob
import os
import pyedflib

#from matplotlib.pyplot import stackplot
#from stacklineplot import stackplot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from multiprocessing.pool import Pool

def stackplot(marray, seconds=None, start_time=None, ylabels=None, ax=None):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    stackplot_t(tarray, seconds=seconds, start_time=start_time, ylabels=ylabels, ax=ax)
    #plt.show()
    #plt.cla()
    #plt.clf()
    #plt.close('all')
    #plt.savefig('fname.png')


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
    #plt.cla()
    #plt.clf()
    #plt.close('all')

def test_stacklineplot():
    numSamples, numRows = 800, 5
    data = np.random.randn(numRows, numSamples)  # test data
    stackplot(data, 10.0)

#if __name__ == '__main__':
    #f = pyedflib.data.test_generator()
    #01_tcp_ar
    #file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/01_tcp_ar/003/00000355/s003_2013_01_04/00000355_s003_t000.edf"
    #file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/01_tcp_ar/018/00001819/s002_2012_04_30/00001819_s002_t000.edf"
    #file = '/Users/mac/Downloads/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/edf/train/01_tcp_ar/000/00000077/s003_2010_01_21/00000077_s003_t000.edf'
    
    #02_tcp_le
    #file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/02_tcp_le/003/00000355/s001_2003_10_14/00000355_s001_t001.edf"
    #03_tcp_ar_a
    #file = "/Users/mac/Downloads/TUH_EEG/edf/epilepsy/03_tcp_ar_a/008/00000883/s003_2010_09_01/00000883_s003_t000.edf"
    
    #01_tcp_ar
    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/01_tcp_ar/027/00002744/s002_2011_04_13/00002744_s002_t001.edf"
    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/01_tcp_ar/066/00006687/s001_2010_06_02/00006687_s001_t000.edf"
    #file = "/Users/mac/Downloads/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/edf/dev/01_tcp_ar/002/00000258/s003_2003_07_22/00000258_s003_t000.edf"
    #02_tcp_le
    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/02_tcp_le/027/00002744/s001_2006_04_13/00002744_s001_t000.edf"
    
    #03_tcp_ar_a
    #file = "/Users/mac/Downloads/TUH_EEG/edf/no_epilepsy/03_tcp_ar_a/076/00007671/s002_2011_02_03/00007671_s002_t000.edf"



def save_stackplots(sourcedir, destdir):    
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*/*/*/*.edf'):
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        n_min = f.getNSamples()[0]
        sigbufs = [np.zeros(f.getNSamples()[i]) for i in np.arange(n)]
        for i in np.arange(n):
            sigbufs[i] = f.readSignal(i)
            if n_min < len(sigbufs[i]):
                n_min = len(sigbufs[i])
        #f._close()
        #del f

        n_plot = np.min((n_min, 2))
        sigbufs_plot = np.zeros((n, n_plot))
        for i in np.arange(n):
            sigbufs_plot[i,:] = sigbufs[i][:n_plot]

        stackplot(sigbufs_plot[:, :n_plot], ylabels=signal_labels)
        plt.savefig(destdir+'/img-'+str(filecnt)+'.png')
        
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

start = time.time()
sourcedir = '/Users/mac/Downloads/TUH_EEG/edf/epilepsy/01_tcp_ar'
destdir   = '/Users/mac/Downloads/TUH_EEG/img/Epilepsy'


os.makedirs(destdir, exist_ok=False)
print("The new directory is created!")
with Pool(28) as p:
    p.map(save_stackplots(sourcedir, destdir))
#save_stackplots(sourcedir, destdir)
end = time.time()
time_to_transform = (end - start)/60
print("Total time (min) for transforming edege :", time_to_transform)
print("=======End transforming edege process here======")