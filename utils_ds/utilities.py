

############################ 
# Imports
############################
import os
# import sys
# import librosa
import logging
# import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
from soundfile import SoundFile



############################ 
# Functions
############################
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def traverse_folder(fd):
    """return a list of all files in a folder that are not png/Thumb and their path 

    Args:
        fd : path to a folder

    Returns:
        names: filename inside the folder 
        paths: respctive path of each file
    """
    paths = []
    names = []

    for root, dirs, files in os.walk(fd, followlinks=True):
        for name in files:
            if name.__contains__('png') or name.__contains__('Thumb'):
                continue
            filepath = os.path.join(root, name)

            names.append(name)
            paths.append(filepath)
    return names, paths


def traverse_folder_withsplit(fd, clip_samples):
    paths = []
    names = []
    offsets = []

    for root, dirs, files in os.walk(fd, followlinks=True):
        for name in files:
            if name.__contains__('png'):
                continue
            if not name.__contains__('wav'):
                continue
            filepath = os.path.join(root, name)
            # print(filepath)
            b = os.path.getsize(filepath)
            myfile = SoundFile(filepath)
            fs = myfile.samplerate
            # print(myfile.subtype)
            sample_w = 8 if myfile.subtype == 'DOUBLE' else 4
            sec_frames = (b // sample_w) // clip_samples
            # if sec_frames > 2 : sec_frames = sec_frames - 3
            for sec in range(sec_frames):
                names.append(name + str(sec))
                paths.append(filepath)
                offsets.append(sec)

    return names, paths, offsets

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def get_metadata(audio_names, audio_paths):
    meta_dict = {
        'audio_name': audio_names, 
        'audio_path': audio_paths, 
        'fold': np.arange(len(audio_names))}

    return meta_dict


def float32_to_int16(x):
    # assert np.max(np.abs(x)) <= 1.
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


############################ 
# Classes
############################
class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'validate': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'validate': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)



############################ 
# Main
############################
def main():

    pass


if __name__=="__main__":
    main()