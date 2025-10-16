import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotterManager:
    def __init__(self, file_name=None, log_data=None):
        plt.style.use('seaborn')
        self.name = file_name

        assert any([file_name, log_data])

        if log_data:
            self.log_data_reference = log_data

        elif file_name:
            self.log_data_reference = None

    @property
    def data(self):
        if self.log_data_reference is not None:
            return self.log_data_reference

        return pd.read_csv(self.name).to_dict()

    @property
    def last_doa_data(self):
        return self._last_data('time', 'doa_circular')

    @property
    def last_classification_data(self):
        return self._last_data('time', 'ml_class', y_str=True)

    @property
    def last_polar_data(self):
        return self._last_data('doa', 'distance', arange=False, data_length=10)

    def _last_data(self, x, y, mic='mic_unit', data_length=60, y_str=False, arange=True):
        c = len(set(self.data[mic]))
        size = max(len(self.data[x]) - (data_length * c), 0)

        y0 = self.data[y][size:]
        s = self.data[mic][size:]
        s_no_arange = np.array(self.data[x][size:])

        out = []

        for m in set(s):
            assert len(y0) == len(s)

            y1 = np.array(y0)
            s1 = np.array(s)

            y1 = y1[s1 == m]
            if arange:
                x1 = np.arange(len(y1))
            else:
                s = np.array(s)
                x1 = s_no_arange[s1 == m]

            if y_str:
                out.append((x1, y1))
            else:
                out.append(np.c_[x1, y1])

        return out

    @property
    def last_beams_amp(self):
        amps = self.data['amp'][-8:]
        beams = self.data['beam'][-8:]
        if len(beams) != 8:
            return np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

        out = np.array([amps, beams])
        out = np.flip(out[:, out[1].argsort()], axis=1)
        return out.reshape([4, 4])[:2, ]

    def __bool__(self):
        return any([self.name, self.log_data_reference])
