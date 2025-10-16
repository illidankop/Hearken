import re

import numpy as np


class AudioBase:
    def __init__(self, samples, rate, t, sns=0):
        self.samples = samples
        self.rate = rate
        self.time = t
        self.sns = sns

    def __getattribute__(self, name):
        # getting the channel data regardless if its size
        if bool(re.match(r'ch\d+_data', name)):
            number = int(re.match(r'ch(\d+)_data', name).group(1))
            if number == 1:
                return self.samples if not self.multi_channel_sample else self.samples[0, :]

            elif self.samples.shape[0] < number - 1:
                raise ValueError("This is a single channel sample, This data does not exists!")

            return self.samples[number - 1, :]
        else:
            return object.__getattribute__(self, name)

    def __len__(self):
        return len(self.ch1_data)

    def data_around_peak(self, peak_loc, frame_time):
        """
        :param: frame_time - frame time in seconds
        """
        assert len(self) / self.rate >= frame_time, "sample dont have enough data for peak frame"

        st_index = int(max(0, peak_loc - (frame_time * self.rate)))
        end_index = int(min(len(self), peak_loc + (frame_time * self.rate)))
        return self.ch1_data[st_index: end_index]

    @property
    def multi_channel_sample(self):
        if not self.samples is None:
            return len(self.samples.shape) - 1
        else:
          return None    

    @property
    def shape(self):
        return self.samples.shape

    @property
    def channel_count(self):
        return self.shape[0]

    def __bool__(self):
        return bool(self.rate)

    def __add__(self, other):
        assert self.rate == other.rate
        assert len(self) == len(other)
        assert self.sns == other.sns
        return __class__(np.hstack([self.samples, other.samples]), self.rate, self.time)
