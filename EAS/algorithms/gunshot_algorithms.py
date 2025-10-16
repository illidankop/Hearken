import numpy as np
# import matplotlib.pyplot as plt
import math
from numpy.core.numeric import rollaxis
from scipy.signal import find_peaks

from .audio_algorithms import gcc_phat

import logging
from eas_configuration import EasConfig
# from pyroomacoustics.doa.srp import SRP
from scipy import signal


config = EasConfig()
# EAS = EasProcessor()
logger = logging.getLogger()


def detect_gunshots_old(shot):
    """
    Detect and measure times for gunshot and blast patterns

    :param data: an audio buffer (float values 0-1) that may contain gun-shot patterns
    :param samp_rate: the audio sample rate
    :param shot: a shot object, contaning samples (normilized) and sample rate
    :return: is_blast - blast is detected True / False
             is_shockwave - shockwave is detected True / False
             b_time - time of blast from start (in samples)
             s_time1 - time of shockwave from start (in samples)
    """

    data, samp_rate = shot.ch1_data, shot.rate
    win_sec = 0.010  # 10 ms
    samp_len = len(data)
    win_size = int(win_sec * samp_rate)
    windows = int(samp_len // win_size) * 2 - 1
    shock_found = False
    shock_time = 0
    is_blast = 0
    blast_time = 0
    for i in range(0, windows):
        start = int(i * win_size // 2)
        end = start + win_size
        samples = data[start:end]
        max_samp_ind = np.argmax(samples)
        min_samp_ind = np.argmin(samples)
        mean_samp = np.mean(samples)
        std_samp = np.std(samples)
        max_samp = samples[max_samp_ind]
        min_samp = samples[min_samp_ind]

        # detect shockwave
        if not shock_found:
            shock_found, shock_time = detect_shock(start, samples, samp_rate)

        # detect blast
        is_blast, blast_time = detect_blast(start, min_samp_ind, max_samp_ind, min_samp, mean_samp, max_samp, samp_rate,
                                            shock_found, shock_time)

        if is_blast and shock_found:
            return is_blast, blast_time, shock_time

    return is_blast, blast_time, shock_time


def detect_blast(start, min_samp_ind, max_samp_ind, min_samp, mean_samp, max_samp, samp_rate, shock_found, shock_time):
    blast = True
    if max_samp < 0.1:
        blast = False

    max_first = (max_samp_ind < min_samp_ind)
    if not max_first:
        blast = False

    # the maximum should be more than 10% above average
    overshoot = abs((max_samp - mean_samp) / mean_samp)
    if overshoot < 2 or overshoot > 400:
        blast = False

    # Y symmetry
    # the distance of the max from the mean should be up to twice the distance of the min from the mean
    val_dist_fact = (max_samp - mean_samp) / (mean_samp - min_samp)

    if max_samp < 0.5 and (val_dist_fact > 2 or val_dist_fact < 0.65):
        blast = False
    # very high power case
    if max_samp > 0.5 and (val_dist_fact > 2.5 or val_dist_fact < 0.9):
        blast = False

    # length 0.8ms to 8ms
    ind_dist = int(min_samp_ind - max_samp_ind)
    time_dist = ind_dist / samp_rate
    if time_dist > 0.008 or time_dist < 0.0008:
        blast = False

    blast_time = _get_blast_time(start, samp_rate, max_samp_ind, ind_dist)

    # blast should be more than 10ms after shock
    if shock_found:
        if (blast_time - shock_time) < 0.01:
            blast = False

    return blast, blast_time


def _get_blast_time(start, samp_rate, max_samp_ind, ind_dist):
    blast_ind = int(max_samp_ind - ind_dist // 2)
    blast_time = (start + blast_ind) / samp_rate
    return blast_time


def detect_shock(start, samples, samp_rate):
    shock_found, shock_time = 0, 0
    peaks, properties = find_peaks(samples, prominence=0.1)
    if peaks.size > 0:
        peak_ind = peaks[0]
        shock_time = (start + peak_ind) / samp_rate
        shock_found = True
    return shock_found, shock_time

def get_gunshot_aoa(event_time, fs, s1, s2, mics_distance):

    CORR_WINDOW = 0.01
    samp1_start = int(event_time * fs) - 50 # take 50 samples before the event start
    samp1 = s1[samp1_start:samp1_start + int(CORR_WINDOW * fs)]
    samp2 = s2[samp1_start:samp1_start + int(CORR_WINDOW * fs)]
    max_tau = mics_distance / config.speed_of_sound

    try:
        ang, cc = gcc_phat(samp1, samp2, fs=fs, max_tau=max_tau, interp=1, dist=mics_distance)

    except ValueError:
        logger.debug("failed to calculate aoa")
        return 0, 0

    return ang


def calculate_source_distance(shock_angle, blast_angle, dt, c=config.speed_of_sound):
    d = (c * dt) / (1 - abs(np.sin(np.deg2rad(shock_angle - blast_angle))))
    return d


def calculate_rough_distance(dt, bullet_v=650, c=config.speed_of_sound):
    return (bullet_v - c) * dt

def get_sw_aoa(s1, s2, rate, f, window_size, mic_dist):
    """
    Calculate the AOA of SW event based on the delay between the channels, which we find using rolling average
    Input: 
    s1, s2 - Signals for channels for comparison
    rate - Audio samping rate
    f - Sensitivity factor
    window_size - Size for the sliding windows for rolling average calculation
    mic_dist - Distance between the mic units
    """
    window= np.ones(int(window_size))/float(window_size)
    ro_av_1 =  np.convolve(abs(s1), window, 'same')
    start_sample_1 = np.where(ro_av_1 > f[0] if f[0] < 0.1 else ro_av_1 > 0.1)
    ro_av_2 =  np.convolve(abs(s2), window, 'same')
    start_sample_2 = np.where(ro_av_2 > f[1] if f[1] < 0.1 else ro_av_2 > 0.1)
    try:
        shift = start_sample_2[0][0] - start_sample_1[0][0]
        dt = shift / rate
        dist = dt * config.speed_of_sound
        aoa = np.degrees(np.arccos(dist / mic_dist))
    except:
        return math.nan, math.nan
    return aoa, shift