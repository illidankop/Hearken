import numpy as np

from .audio_base import AudioBase


class AudioShot(AudioBase):
    mini_shot_len = 0.1

    def __init__(self, samples, rate, t):
        super(AudioShot, self).__init__(samples, rate, t)

    def __iter__(self):
        """
        returns a Shot in a smaller length
        """
        frame_size = int(self.rate * self.mini_shot_len)
        frame_count = int(len(self) // frame_size)
        for s in range(frame_count):

            if self.multi_channel_sample:
                samples = self.samples[:, s * frame_size: (s + 1) * frame_size]
                yield AudioShot(samples, self.rate, self.time + s * self.mini_shot_len)
            else:
                samples = self.samples[s * frame_size: (s + 1) * frame_size]
                yield AudioShot(samples, self.rate, self.time + s * self.mini_shot_len)


    def __repr__(self):
        return f"Shot(sample count: {len(self.ch1_data)},rate: {self.rate},time: {self.time}, ch: {self.channel_count})"

    @property
    def shots_pair_h(self):
        if self.channel_count == 1:
            raise ValueError("no pair in mono mic")

        if self.channel_count == 2:
            return [self]

        l_data = np.vstack([self.samples[self.channel_count - 2, :], self.samples[0, :]])
        out = [AudioShot(self.samples[i:i + 2, :], self.rate, self.time) for i in range(0, self.channel_count - 2)]
        out.append(AudioShot(l_data, self.rate, self.time))
        return out
        
    def __add__(self, other):
        assert self.rate == other.rate
        assert len(self) == len(other)
        assert self.sns == other.sns
        return __class__(np.hstack([self.samples, other.samples]), self.rate, self.time)