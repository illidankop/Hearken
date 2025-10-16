from threading import Thread
from contextlib import contextmanager
from .AAMic import AAMic
from .conf import Configuration
import time

from EAS.proccesers.eas_aa_procceser import AAProcessor


@contextmanager
def streamer(v: AAMic):
    time.sleep(0.1)
    v.start()
    yield
    # time.sleep(3)
    v.terminate()


class AAMain(Thread):
    def __init__(self, sample_rate=48000):
        super(AAMain, self).__init__()
        self.sample_rate = sample_rate
        self.config = Configuration()
        self.p = AAProcessor(sample_rate)
        self.mic = AAMic(sample_rate=sample_rate)
        self.name = 'AA Runner'
        self.alive = True
        self.time_to_write = 60
        self.c = 0

    def run(self):
        self.mic.start()
        while self.alive:
            fr = self.mic.get_frame(block=False)
            if fr[0] is not None:
                self.p.process_frame(fr)

                if self.c and not self.c % self.time_to_write:
                    self.mic.write_to_file()
                self.c += 1

    def join(self, timeout=None) -> None:
        self.alive = False
        self.p.stop()
        self.mic.terminate()
        super(AAMain, self).join(timeout=timeout)
