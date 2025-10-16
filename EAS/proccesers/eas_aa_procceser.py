import numpy as np
from pathlib import Path
import sys
import os
path = Path(os.getcwd())
sys.path.append('{}'.format(path))
sys.path.append('{}'.format(path.parent))
sys.path.append('{}'.format(path.parent.parent))

from proccesers.eas_processing import EasProcessor


class AAProcessor(EasProcessor):
    base_file_name = 'live_AA_results'

    def __init__(self, sample_rate):
        super(AAProcessor, self).__init__()
        self.logging_graph_data = {'time': [], 'class': [], 'certainty': [], 'beam': [], 'amp': []}
        self.sample_rate = sample_rate
        self.c = 0

    def process_frame(self, fr):
        self.logger.info(f"processing IA frame {fr.time}")            
        for c in range(8):
            ml_class, score = self.ml_classifier.classify_from_stream(fr.data[c, :], self.sample_rate)
            assert type(score)==list, "score isnt provided in  a list format"
            self.logging_graph_data['time'].append(fr.time)
            self.logging_graph_data['class'].append(ml_class)
            self.logging_graph_data['certainty'].append(score[0])
            self.logging_graph_data['beam'].append(c)
            self.logging_graph_data['amp'].append(np.sqrt(np.mean(np.power(fr.data[c, :], 2))))

        if self.c % 10 == 0:
            self._write_results()
        self.c += 1

    def stop(self):
        self._write_results()
