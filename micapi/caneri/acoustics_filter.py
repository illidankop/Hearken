# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:51:41 2025

@author: 60407
"""

import numpy as np
import pywt


def buffer_py(X, n, overlap = 0):
    """
    Parameters
    ----------
    X : np.array 1D
        the array to reshape
    n : int
        window size (rows).
    overlap : float, optional
        number of overlap samples (not percentage!).

    Returns
    -------
    output : np.array 2D
        buffered array 

    """
    
    # Calculate the number of columns in the output matrix
    numcol = np.ceil((len(X)-overlap)/(n-overlap)).astype(int)
    output = np.zeros((n, numcol))
    start  = 0
    end    = n
    
    for i in range(0, numcol):
        wind = X[start:end]
        output[0:len(wind),i] = wind
        start = end-overlap
        end   = start+n
    return output


# Noise estimators 
def sqtwolog_method(coeffs):
    """ 
    The threshold value is given by the squared-root of 2 x log(m), where m is 
    the length of the input signal.

    Parameters:
        coeffs (list of np.ndarray):  Detailed coefficients of the wavelet transform.
    Returns:
        threshold (np.ndarray): The threshold value.
    """
    threshold = [np.std(c) * np.sqrt(2 * np.log(len(c))) for c in coeffs]
    return threshold


def calculate_BG_noise(dataset, sample_rate, num_of_channels, num_of_calculated_BG, wavelet):
    max_level = (num_of_calculated_BG-1)        
    all_chnnels_BG = np.zeros((num_of_calculated_BG, num_of_channels),dtype=np.float32)
    for ch in range(3):
        x          = dataset[:, ch]                                
        # Step 2.1: Estimate the BG noise from the first 5 seconds 
        # TODO: in the real-time process, it should be incorporated differently
        x_obs     = x[:int(5*sample_rate)]
        coeffs    = pywt.wavedec(x_obs, wavelet, level=max_level)
        c_std     = sqtwolog_method(coeffs)        
        all_chnnels_BG[:,ch] = c_std
    return all_chnnels_BG


def filter_BG_noise(acoustic_arr, sample_frequency, all_channels_avarage_BG_noise, num_of_calculated_BG, wavelet, soft, window):
    """ 
    The function takes a signal from three microphones, buffers it into segments 
    of interest (1 sec of 50 ms) and filters according to the first 5 sec in the recording

    Parameters:
        acoustic_arr (np.ndarray [N x 3]):  Signal from the three microphones [Pa] 
        sample_frequency (int):                Sample frequency [Hz] (32 kHz in Thunderbolt)
        all_channels_avarage_BG_noise: BG noise calculation for each channel
        wavelet:
        soft: filter soft
        windows: the frmae window size to apply the filter
        
    Returns:
        signal_recon (np.ndarray [N x 3]): Filtered signal from the three microphones [Pa] 
    """
    
    max_level = (num_of_calculated_BG-1)      
    #%% Step 2: Filter each mic accordingly in the loop
    signal_recon = [] # signal variable for saving filtered wav
    
    for ch in np.arange(3):
        x          = acoustic_arr[:, ch]        
        process_window = int(window*sample_frequency)        
        x_buffered = buffer_py(x, process_window) 
        x_recon    = []
        
        # # Step 2.1: Estimate the BG noise from the first 5 seconds 
        # # TODO: in the real-time process, it should be incorporated differently
        # x_obs     = x[:int(5*Fs)]
        # coeffs    = pywt.wavedec(x_obs, w, level=max_level)
        # one_ch_BG_noise     = sqtwolog_method(coeffs)
        
        one_ch_BG_noise = all_channels_avarage_BG_noise[:,ch]
        for isec in np.arange(x_buffered.shape[1]):          
            # Step 2.2: Derive wavelet coefficients
            coeffs = pywt.wavedec(x_buffered[:,isec], wavelet, level=max_level)
            
            # Step 2.3: Apply thresholding for the coeeficients related to the noise std, except for the approximate coefficients 
            filtered_coeffs = [pywt.threshold(c, value=soft*one_ch_BG_noise[i], mode='soft') if i != 0 else c for i, c in enumerate(coeffs)]
            #set nan value to zero - relevant for dead channels
            filtered_coeffs = [np.nan_to_num(coeff, nan=0.0) for coeff in filtered_coeffs]
            # Step 2.4: Inverse wavelet transform of the segment
            x_recon.append(pywt.waverec(filtered_coeffs, wavelet))
        
        # If the data were buffered, complile together
        x_recon_flat = np.array(x_recon).flatten()
        x_recon      = x_recon_flat  
        
        signal_recon.append(x_recon) 
    signal_recon = np.array(signal_recon).T
    
    return signal_recon