import pyedflib
import numpy as np
import glob
import os
from multiprocessing.pool import Pool
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
# use scipy.signal spectrogram to extract raw data
from scipy.signal import spectrogram

def convert(sourcedir, destdir):
    print("\n\n----reading files in directories " + sourcedir + "----\n")
    filecnt = 1
    for file in glob.glob(sourcedir + "/*/*/*/*/*.edf"):
        f = pyedflib.EdfReader(file)
        a = (f.readSignal(2))
        print(a.shape)
        raw = spectrogram(a[:750000], fs=125, noverlap=1)[2]
        im = specgram(a[:750000], Fs=125, noverlap=1)[3]
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.savefig(destdir + '/img_' + str(filecnt) + ".png", bbox_inches = 'tight', pad_inches = 0)
        #f._close()
        filecnt += 1
    print('\n\n--saved in ' + destdir + '---\n')

start = time.time()

sourcedir = "/Users/mac/Downloads/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_epilepsy/v1.0.0/edf/no_epilepsy/"
destdir   = "/Users/mac/Downloads/TUH_EEG/No_Epilepsy"

os.makedirs(destdir, exist_ok=True)
print("The new directory is created")
with Pool(28) as p:
    p.map(convert(sourcedir, destdir))

end = time.time()
time_to_convert = (end - start)/60
print("Total time (min) for converting EEG to images: ", time_to_convert)
print("=======End converting EEG to images here=========")