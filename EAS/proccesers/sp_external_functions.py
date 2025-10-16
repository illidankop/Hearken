import numpy as np
import random

def generate_beams(az_el_grid, last_frame, current_frame, mics):
    fft_size = 4096    
    fft_grid = np.zeros((az_el_grid.shape[0], fft_size),dtype=np.float32)
    return fft_grid


def get_detections_grid(fft_grid, size):
    lst = []
    for _ in range(size):
        lst.append(random.random())
    detection_grid = np.array(lst)
    return detection_grid.T

#def get_calculated_az_el(fft_full_grid, az_el_grid, index):
def get_calculated_az_el(fft_full_grid, azimuth, elevation):
    return azimuth, elevation
