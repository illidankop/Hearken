from .audio_base import AudioBase


class AudioDrone(AudioBase):
    def __init__(self, samples, rate, t, duration = 1):
        super(AudioDrone, self).__init__(samples, rate, t)
        self.duration_in_seconds = duration
