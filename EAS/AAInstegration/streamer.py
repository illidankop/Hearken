from threading import Thread
import time
import numpy as np


class AudioStream(Thread):
    def __init__(self, data_list, sample_rate):
        super(AudioStream, self).__init__()
        self.alive = True
        self.data_list = data_list
        self.sample_rate = sample_rate
        self.name = "AudioStreamer"

    def _play(self, data):
        import simpleaudio as sa
        play_obj = sa.play_buffer(data, 1, 2, self.sample_rate)
        return

    def run(self) -> None:
        c = 0
        while self.alive:
            if self.data_list:
                self._play(self.data_list.popleft())
                continue
            c += 1
            time.sleep(0.001)

    def join(self) -> None:
        self.alive = False
        super(AudioStream, self).join()
