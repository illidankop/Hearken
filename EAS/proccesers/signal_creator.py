import numpy as np
from scipy.signal import chirp


def create_signal(fs, freq, signal_restart, chirp_duration, hold_on):
    hold_on_samples_count = int(hold_on * fs)
    chirp_samples_count = int(chirp_duration * fs)
    signal_restart_samples_count = int((signal_restart - chirp_duration - hold_on) * fs)

    hold_on_samples = np.linspace(0, hold_on, hold_on_samples_count, endpoint=False)
    signal_start = np.int16(np.sin(2 * np.pi * freq * hold_on_samples) * 32767)

    times = np.linspace(0, chirp_duration, chirp_samples_count, endpoint=False)
    chirp_samples = np.int16(
        chirp(times, f0=freq, f1=freq + 2000, t1=chirp_duration - hold_on, method='linear') * 32767)

    # blackout_samples = np.linspace(0, signal_restart - chirp_duration - hold_on, signal_restart_samples_count,
    #                                endpoint=False)
    # blackout = np.int16(np.sin(2 * np.pi * freq * blackout_samples) * 32767)
    # res = np.hstack([signal_start, chirp_samples, blackout])
    res = np.hstack([signal_start, chirp_samples])
    return res


def _create_constant_wave(freq, duration, bitrate):
    points = int(bitrate * duration)
    times = np.linspace(0, duration, points, endpoint=False)
    data = np.int16(np.sin(2 * np.pi * freq * times) * 32767)
    return data


def create_stairs_signal(base_frequency, jump_duration, jump, jump_count, bitrate=48000):
    data = [_create_constant_wave(base_frequency + (jump * i), jump_duration, bitrate) for i in range(jump_count)]
    return np.hstack(data)


def create_signal_generator(fs, freq, signal_restart, chirp_duration, hold_on, d, sample_count=1024, c=343, m_sec=60):
    # this function isnt responseable on start time!

    base = create_signal(fs, freq, signal_restart, chirp_duration, hold_on)

    roll_count = int((d / c) * fs)
    base = np.roll(base, roll_count)

    counter = 0
    while counter * (sample_count / fs) < m_sec:
        yield base[:sample_count]
        counter += 1
        base = np.roll(base, -sample_count)
