# Wearable sensor dataset.

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchaudio.transforms import Spectrogram

import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

import wfdb
import ast
#from src.datasets.root_paths import DATA_ROOTS
DATA_ROOTS = '/users/mac/Downloads/ECG/PTB_XL/'



# DIAGNOSTIC_SUPERCLASS=['NORM','MI','STTC','CD','HYP']
DIAGNOSTIC_SUBCLASS=['ISCA', 'LVH', 'IMI', 'CLBBB', 'LAO/LAE', 'AMI', 'LAFB/LPFB', 'RAO/RAE', 'ISCI', 'NST_', 'NORM', 'PMI', 'IRBBB', 'RVH', 'IVCD', 'LMI', 'CRBBB', 'STTC', '_AVB', 'ILBBB', 'WPW', 'ISC_', 'SEHYP']

FEATURE_MEANS=np.array([-0.00074703,  0.00054328,  0.00128943,  0.0001024 , -0.00096791,
        0.00094267,  0.0008255 , -0.00062468, -0.00335543, -0.00189922,
        0.00095845,  0.000759  ])

FEATURE_STDS=np.array([0.13347071, 0.19802795, 0.15897414, 0.14904783, 0.10836737,
       0.16655428, 0.17850298, 0.33520913, 0.28028072, 0.27132468,
       0.23750131, 0.19444742])


class PTB_XL(data.Dataset):
    NUM_CLASSES = 23  # NOTE: They're not contiguous labels.
    NUM_CHANNELS = 12 # Multiple sensor readings from different parts of the body
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
        self,
        mode='train',
        sensor_transforms= 'spectral_noise', #None,
        root=DATA_ROOTS, #['ptb_xl'],
        examples_per_epoch=10000  # Examples are generated stochastically.
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.sensor_transforms = sensor_transforms
        self.dataset = BasePTB_XL(
            mode=mode, 
            root=root, 
            examples_per_epoch=examples_per_epoch)
    
    def transform(self, spectrogram):
        if self.sensor_transforms:
            if self.sensor_transforms == 'spectral':
                spectral_transforms = SpectrumAugmentation()
            elif self.sensor_transforms == 'spectral_noise':
                spectral_transforms = SpectrumAugmentation(noise=True)
            elif self.sensor_transforms == 'just_time':
                spectral_transforms = SpectrumAugmentation(just_time=True)
            else:
                raise ValueError(f'Transforms {self.sensor_transforms} not implemented.')

            spectrogram = spectrogram.numpy().transpose(1, 2, 0)
            spectrogram = spectral_transforms(spectrogram)
            spectrogram = torch.tensor(spectrogram.transpose(2, 0, 1))
        elif self.sensor_transforms:
            raise ValueError(
                f'Transforms "{self.sensor_transforms}" not implemented.')
        return spectrogram

    def __getitem__(self, index):
        # pick random number
        img_data, label = self.dataset.__getitem__(index)
        subject_data = [
            index,
            self.transform(img_data).float(), 
            self.transform(img_data).float(),
            label]

        return tuple(subject_data)

    def __len__(self):
        return self.examples_per_epoch



class BasePTB_XL(data.Dataset):

    def __init__(
        self,
        mode='train',
        root=DATA_ROOTS, #['ptb_xl'],
        measurements_per_example=1000,
        examples_per_epoch=10000,
        normalize=True
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.measurements_per_example = measurements_per_example  # Measurements used to make spectrogram
        self.mode = mode
        self.subject_data = self.load_data(root)
        self.normalize = normalize

    def get_subject_ids(self, mode):
        if mode == 'train':
            nums = [1,2,3,4,5,6,7,8]
        elif mode == 'train_small':
            nums = [1]
        elif mode == 'val':
            nums = [9]
        elif mode == 'test':
            nums = [10]
        else:
            raise ValueError(f'mode must be one of [train, train_small, val, test]. got {mode}.')
        return nums  

    def load_data(self, root_path):
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        sampling_rate=100

        # load and convert annotation data
        print("load and convert annotation data")
        Y = pd.read_csv(root_path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        print("Load raw signal data")
        X = load_raw_data(Y, sampling_rate, root_path)

        # Load scp_statements.csv for diagnostic aggregation
        print("Load scp_statements.csv for diagnostic aggregation")
        agg_df = pd.read_csv(root_path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_subclass)
            conf=list(y_dic.values())
            inds = []
            seen = set()
            for i, ele in enumerate(tmp):
                if ele not in seen:
                    inds.append((i))
                seen.add(ele)
            tmp1=[tmp[i] for i in inds]
            conf1=[conf[i] for i in inds]
            return  tmp1,conf1
        Y['diagnostic_subclass'], Y['diagnostic_confidence'] = zip(*Y.scp_codes.apply(aggregate_diagnostic))

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        #         print("X train shape:", X_train.shape)
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_subclass
        #         print("y data", y_train)
        y_conf= Y[(Y.strat_fold != test_fold)].diagnostic_confidence.to_numpy()
        y_train = y_train.to_numpy()
        #         print("y train test", y_train[0][0])
        #         print("y train shape:", y_train.shape)
        #         print()
        subject_data=[X_train,y_train,y_conf]
        return subject_data
    
    def __getitem__(self, index):
        while True:
            ecgid = np.random.randint(len(self.subject_data[0]))
            if len(self.subject_data[1][ecgid]) > 0: break
                
#         print("example diagnosis id", self.subject_data[1][ecgid])
        
        max_conf=np.argmax(self.subject_data[2][ecgid])
        diagnosis_id = DIAGNOSTIC_SUBCLASS.index(self.subject_data[1][ecgid][max_conf])
        measurements = self.subject_data[0][ecgid]

        # Yields spectrograms of shape [52, 32, 32]
        spectrogram_transform=Spectrogram(n_fft=64-1, hop_length=32, power=2)
        spectrogram = spectrogram_transform(torch.tensor(measurements.T))
        spectrogram = (spectrogram + 1e-6).log()
        if self.normalize:
            spectrogram = (spectrogram - FEATURE_MEANS.reshape(-1, 1, 1)) / FEATURE_STDS.reshape(-1, 1, 1)
#         print("spectrogram shape", spectrogram.shape)
#         print("diagnosis_id", diagnosis_id)
        return spectrogram, diagnosis_id

    
    def __len__(self):
        return self.examples_per_epoch


class SpectrumAugmentation(object):

    def __init__(self, just_time=False, noise=False):
        super().__init__()
        self.just_time = just_time
        self.noise = noise

    def get_random_freq_mask(self):
        return nas.FrequencyMaskingAug(mask_factor=20)

    def get_random_time_mask(self):
        return nas.TimeMaskingAug(coverage=0.7)

    def __call__(self, data):
        if self.just_time:
            transforms = naf.Sequential([self.get_random_time_mask()])
        else: 
            transforms = naf.Sequential([self.get_random_freq_mask(),
                                     self.get_random_time_mask()])
        data = transforms.augment(data)
        if self.noise:
            noise_stdev = 0.25 * np.array(FEATURE_STDS).reshape(1, 1, -1)
            noise = np.random.normal(size=data.shape) * noise_stdev
            data = data + noise
        return data