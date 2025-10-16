import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
import random
from utilities import int16_to_float32

import librosa
import torch
from torch.utils.data import Dataset
import scipy

############################ 
# Dataset
###########################

def datasetFactory(expType):
    """Get a name of experiment and return the corresponding dataset

    Args:
        type (str): experiment type from args.yaml 

    Returns:
        torch dataset
    """
    return eval(expType + 'Dataset')


class  A_NBDataset(Dataset):
    def __init__(self, clip_samples,hdf5_path, use_folds, transform=None):        
        """ Torch dataset, return samples of audio name,waveform and label with respect to the given folds
            Classes: ATMs vs else (nonAtms + Background) 
        Args:
            clip_samples (_type_): deprecated
            hdf5_path (str): path to h5 dataset file 
            use_folds (list<int>): which folds to sample from
            transform ( optional): augmentations/transform to execute on each sample before returning it, Defaults to None.
        """
        self.clip_samples = clip_samples
        self.hdf5_path = hdf5_path 
        self.transform = transform
        self.use_folds = use_folds

        try:
            self.h5_data = h5py.File(self.hdf5_path,'r')
        except:
            print(f"Can`t open hdf5 file {self.hdf5_path}")


        self.folds = self.h5_data['fold'][:].astype(np.float32)
        if(use_folds is None):
            self.use_folds = np.unique(self.folds)
        indices = np.array([False]*len(self.folds))
        for f in self.use_folds:
            indices = (self.folds == int(f)) | indices
        
        self.indices = np.where(indices)[0]
        

    def __len__(self):
        return  len(self.indices)
    
    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        index = self.indices[idx]
        audio_name = self.h5_data['audio_name'][index].decode()
        waveform = int16_to_float32(self.h5_data['waveform'][index])
        target = self.h5_data['target'][index].astype(np.float32)
        
        if(target[1]==1 or target[2]==1):
            target = np.array([0.,1.])
        else:
            target = np.array([1.,0.])

        if self.transform:
                waveform = self.transform(waveform) 

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict

class  A_B_NDataset(Dataset):
    def __init__(self, clip_samples,hdf5_path, use_folds, transform=None):
        """ Torch dataset, return samples of audio name,waveform and label with respect to the given folds
            Classes: ATMs vs nonAtms vs Background
        Args:
            clip_samples (_type_): deprecated
            hdf5_path (str): path to h5 dataset file 
            use_folds (list<int>): which folds to sample from
            transform ( optional): augmentations/transform to execute on each sample before returning it, Defaults to None.
        """
        self.clip_samples = clip_samples
        self.hdf5_path = hdf5_path 
        self.transform = transform
        self.use_folds = use_folds

        try:
            self.h5_data = h5py.File(self.hdf5_path,'r')
        except:
            print(f"Can`t open hdf5 file {self.hdf5_path}")


        self.folds = self.h5_data['fold'][:].astype(np.float32)
        if(use_folds is None):
            self.use_folds = np.unique(self.folds)
            
        indices = np.array([False]*len(self.folds))
        for f in self.use_folds:
            indices = (self.folds == int(f)) | indices
        
        self.indices = np.where(indices)[0]
        

    def __len__(self):
        return  len(self.indices)
    
    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        index = self.indices[idx]
        audio_name = self.h5_data['audio_name'][index].decode()
        waveform = int16_to_float32(self.h5_data['waveform'][index])
        target = self.h5_data['target'][index].astype(np.float32)

        if self.transform:
            for t in self.transform:
                waveform = t(waveform) # Chose class transform based on y

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict

class  AN_BDataset(Dataset):
    def __init__(self, clip_samples,hdf5_path, use_folds, transform=None):
        """ Torch dataset, return samples of audio name,waveform and label with respect to the given folds
            Classes: Explosion(ATMS + nonATMs) vs Background 
        Args:
            clip_samples (_type_): deprecated
            hdf5_path (str): path to h5 dataset file 
            use_folds (list<int>): which folds to sample from
            transform ( optional): augmentations/transform to execute on each sample before returning it, Defaults to None.
        """
        self.clip_samples = clip_samples
        self.hdf5_path = hdf5_path 
        self.transform = transform
        self.use_folds = use_folds

        try:
            self.h5_data = h5py.File(self.hdf5_path,'r')
        except:
            print(f"Can`t open hdf5 file {self.hdf5_path}")


        self.folds = self.h5_data['fold'][:].astype(np.float32)
        if(use_folds is None):
            self.use_folds = np.unique(self.folds)      

        indices = np.array([False]*len(self.folds))
        for f in self.use_folds:
            indices = (self.folds == int(f)) | indices
        
        self.indices = np.where(indices)[0]
        

    def __len__(self):
        return  len(self.indices)
    
    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        index = self.indices[idx]
        audio_name = self.h5_data['audio_name'][index].decode()
        waveform = int16_to_float32(self.h5_data['waveform'][index])
        target = self.h5_data['target'][index].astype(np.float32)
        
        if(target[1]==1 or target[0]==1):
            target = np.array([1.,0.])
        else:
            target = np.array([0.,1.])
        
        if self.transform:
            for t in self.transform:
                waveform = t(waveform) # Chose class transform based on y
        

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict

class  BackgroundDataset(Dataset):
    def __init__(self, clip_samples,hdf5_path, use_folds, transform=None):
        """ Torch dataset, return samples of audio name,waveform and label with respect to the given folds
            This dataset returns only Background samples
        Args:
            clip_samples (_type_): deprecated
            hdf5_path (str): path to h5 dataset file 
            use_folds (list<int>): which folds to sample from
            transform ( optional): augmentations/transform to execute on each sample before returning it, Defaults to None.
        """
        self.clip_samples = clip_samples
        self.hdf5_path = hdf5_path 
        self.transform = transform
        self.use_folds = use_folds

        try:
            self.h5_data = h5py.File(self.hdf5_path,'r')
        except:
            print(f"Can`t open hdf5 file {self.hdf5_path}")


        self.folds = self.h5_data['fold'][:].astype(np.float32)
        if(use_folds is None):
            self.use_folds = np.unique(self.folds)
            
        indices = np.array([False]*len(self.folds))
        for f in self.use_folds:
            indices = (self.folds == int(f)) | indices
        
        indices = indices & (self.h5_data['target'][:].astype(np.float32)[:,1]==1)

        self.indices = np.where(indices)[0]
        

    def __len__(self):
        return  len(self.indices)
    
    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        index = self.indices[idx]
        audio_name = self.h5_data['audio_name'][index].decode()
        waveform = int16_to_float32(self.h5_data['waveform'][index])
        target = self.h5_data['target'][index].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict

class  A_NDataset(Dataset):
    def __init__(self, clip_samples,hdf5_path, use_folds, transform=None):
        """ Torch dataset, return samples of audio name,waveform and label with respect to the given folds
            Classes: ATMs vs nonATMs
        Args:
            clip_samples (_type_): deprecated
            hdf5_path (str): path to h5 dataset file 
            use_folds (list<int>): which folds to sample from
            transform ( optional): augmentations/transform to execute on each sample before returning it, Defaults to None.
        """
        self.clip_samples = clip_samples
        self.hdf5_path = hdf5_path 
        self.transform = transform
        self.use_folds = use_folds

        try:
            self.h5_data = h5py.File(self.hdf5_path,'r')
        except:
            print(f"Can`t open hdf5 file {self.hdf5_path}")


        self.folds = self.h5_data['fold'][:].astype(np.float32)
        if(use_folds is None):
            self.use_folds = np.unique(self.folds)
        indices = np.array([False]*len(self.folds))
        for f in self.use_folds:
            indices = (self.folds == int(f)) | indices
        
        indices = indices & (self.h5_data['target'][:].astype(np.float32)[:,1]==0)

        self.indices = np.where(indices)[0]
        

    def __len__(self):
        return  len(self.indices)
    
    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        index = self.indices[idx]
        audio_name = self.h5_data['audio_name'][index].decode()
        waveform = int16_to_float32(self.h5_data['waveform'][index])
        target = self.h5_data['target'][index].astype(np.float32)
        if(target[1]==1 or target[2]==1):
            target = np.array([0.,1.])
        else:
            target = np.array([1.,0.])

        if self.transform:
                waveform = self.transform(waveform) 

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict


############################ 
# Functions
###########################
def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict

def normalize(x,a,b):
    if np.min(x) == np.max(x):
        return x
    return ((b-a)*(x - np.min(x))/(np.max(x) - np.min(x))) + a

############################ 
# Augmentations
###########################

class MixBackground(torch.nn.Module):
    def __init__(self,backgroundSet, mb_alpha=0.2,prob=0.2) -> None:
        super().__init__()
        self.backgroundSet = backgroundSet
        self.mb_alpha = mb_alpha
        self.prob = prob

    def forward(self,x):
        if random.random() < self.prob:
            idx = np.random.randint(0,len(self.backgroundSet))
            y =  self.mb_alpha * self.backgroundSet[idx]['waveform'] + x
            # y = normalize(y,-1,1)
            return y
        return x

class WhiteNoise(torch.nn.Module):
    def __init__(self,noiseFactor = 0.3, prob=0.5) -> None:
        super().__init__()
        self.prob = prob
        self.noiseFactor = noiseFactor

    def forward(self,x):
        if random.random() < self.prob:
            y = x + self.noiseFactor*np.random.normal(0,0.1,size=( len(x)),)
            # y = normalize(y,-1,1)
            return y
        return x

# class PitchShiftOld(torch.nn.Module):
#     def __init__(self, sr, pitchFactor, prob=0.5) -> None:
#         super().__init__()
#         self.sr = sr
#         self.pitchFactor = pitchFactor
#         self.prob = prob
#     def forward(self,x):
#         if random.random() < self.prob:
#             return librosa.effects.pitch_shift(y=x, sr=self.sr, n_steps=self.pitchFactor)
#         return x

class PitchShift(torch.nn.Module):
    def __init__(self, sr, pitchDnMin, pitchDnMax, pitchUpMin, pitchUpMax, prob=0.5) -> None:
        super().__init__()
        self.sr = sr
        self.pitchDnMin = pitchDnMin
        self.pitchDnMax = pitchDnMax
        self.pitchUpMin = pitchUpMin
        self.pitchUpMax = pitchUpMax
        self.prob = prob

    def forward(self, x):
        # print('pitch_input.shape=',x.shape)
        if random.random() < self.prob:
            # Combine the ranges for pitch down and pitch up, ensuring we exclude 0
            pitchFactors = list(range(self.pitchDnMin, self.pitchDnMax + 1)) + list(range(self.pitchUpMin, self.pitchUpMax + 1))
            pitchFactor = random.choice(pitchFactors)
            x_pitched = librosa.effects.pitch_shift(y=x, sr=self.sr, n_steps=pitchFactor)
            # print('pitch_output.shape=',x_pitched.shape)
            return x_pitched
            #return librosa.effects.pitch_shift(y=x.numpy(), sr=self.sr, n_steps=pitchFactor)
            # return librosa.effects.pitch_shift(y=x, sr=self.sr, n_steps=pitchFactor)
        return x
    
class TimeStretch(torch.nn.Module):
    def __init__(self, stretchMin=0.75, stretchMax=1.25, prob=0.5) -> None:
        """
        Initializes the time stretch augmentation module.
        Args:
            stretchMin (float): Minimum stretch factor. Default is 0.75, which corresponds to a 25% decrease in speed.
            stretchMax (float): Maximum stretch factor. Default is 1.25, which corresponds to a 25% increase in speed.
            prob (float): Probability with which the augmentation is applied. Default is 0.5.
        """
        super().__init__()
        self.stretchMin = stretchMin
        self.stretchMax = stretchMax
        self.prob = prob

    def forward(self, x):
        """
        Applies time stretch augmentation to the input audio.
        Args:
            x (Tensor): Input audio tensor.
        Returns:
            Tensor: Augmented audio tensor.
        """
        if random.random() < self.prob:
            print('stretch_input.shape=',x.shape)
            stretchFactor = random.uniform(self.stretchMin, self.stretchMax)
            x_stretched = librosa.effects.time_stretch(y=x, rate=stretchFactor)
            print('stretch_outut.shape=',x_stretched.shape)
            return torch.from_numpy(x_stretched)
            # return librosa.effects.time_stretch(y=x, rate=stretchFactor)
        return x
    
class PolarityInversion(torch.nn.Module):
    def __init__(self, prob=0.5) -> None:
        super().__init__()
        self.prob = prob
    def forward(self,x):
        if random.random() < self.prob:
            return -1*x
        return x

class MovingAverage(torch.nn.Module):
    def __init__(self, sr, cutOff = 16000, prob=0.1) -> None:
        super().__init__()
        self.prob = prob
        self.cutOff = cutOff
        self.sr = sr

        freqRatio = self.cutOff / self.sr
        self.window_length = int(np.sqrt(0.196201 + freqRatio**2) / freqRatio)
        self.window = np.ones(self.window_length)/self.window_length

    def forward(self,x):
        if random.random() < self.prob:
            return scipy.signal.lfilter(self.window, [1], x)

        return x

class Resample(torch.nn.Module):
    def __init__(self, sr, new_sr=16000, prob=1) -> None:
        super().__init__()
        self.prob = prob
        self.sr = sr
        self.new_sr=new_sr

    def forward(self,x):
        if(self.sr==self.new_sr):
            return x
        elif random.random() < self.prob:
            # return librosa.resample(x, orig_sr=self.sr, target_sr=self.new_sr)
            return scipy.signal.resample(x, self.new_sr, t=None, axis=0, window=None, domain='time')
        return x


############################ 
# Guard
###########################
import yaml
from box import Box
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))  # Remove?
from model_config import Config

if __name__=='__main__':
    args = yaml.load(open("args.yaml"), Loader=yaml.FullLoader)
    # Dot notation
    args = Box(args)

    expType = 'A_B_N'
    config = Config(args[expType])
    a = A_B_NDataset(config.clip_samples,'/home/daniel/Documents/repos2/testTrain/ATMDetection/datasets/atmtypes_noTest_out.h5',use_folds=None)
   
    for i in range(len(a)):
        if('BK' in a[i]['audio_name'] and np.argmax(a[i]['target'])!=1):
            print(f"file:{a[i]['audio_name']}")
            print(f"target:{a[i]['target']}")
   
    pass