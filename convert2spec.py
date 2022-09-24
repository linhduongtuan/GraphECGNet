import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

fpath = '/Users/mac/Downloads/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_epilepsy/v1.0.0/edf/epilepsy/01_tcp_ar/003/00000355/s003_2013_01_04/00000355_s003_t000.edf'
f = mne.io.read_raw_edf(fpath)
events = mne.read_events(f)
#sample_data_folder = '/Users/mac/Downloads/isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_epilepsy/v1.0.0/edf/epilepsy/01_tcp_ar/003/00000355/s003_2013_01_04/'
#sample_data_folder = mne.datasets.sample.data_path()
#sample_data_raw_file = os.path.join(sample_data_folder, '00000355_s003_t000.edf')
#raw = mne.io.read_raw_edf(sample_data_raw_file, preload=False)

#sample_data_events_file = os.path.join(sample_data_folder, '00000355_s003_t000.edf')
#events = mne.read_events(sample_data_events_file)

raw.crop(tmax=90)  # in seconds; happens in-place
# discard events >90 seconds (not strictly necessary: avoids some warnings)
events = events[events[:, 0] <= raw.last_samp]