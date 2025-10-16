import sys
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from EAS.eas_main import EaServer
from EAS.ui import Ui_Dialog


class MainWindow(Ui_Dialog):
    def __init__(self, ac: EaServer):
        super(MainWindow).__init__()
        self.ac = ac

    def setupUi(self, Dialog):
        self.d = Dialog

        super(MainWindow, self).setupUi(Dialog)
        self._reset_beam_values()

        # functions
        self.update_values.clicked.connect(self._set_values)
        self.reset_values.clicked.connect(self._reset_beam_values)
        self.save_audio_button.clicked.connect(self.ac.save_audio)
        self.terminate.clicked.connect(self._terminate)

        # graphs
        self.canvas = []
        self.doa_scatters = []
        self.ml_scatters = []
        self.amp_scatters = []
        self.polar_scatters = []
        self.timers = []
        self.amps = []
        self.figures = []

        self._create_canvas(self.tab, self.doa_scatters, self._update_canvas_doa, add_toolbar=False)
        self._create_canvas(self.tab_2, self.ml_scatters, self._update_canvas_ml, add_toolbar=False, set_lim=False)
        self._create_canvas(self.BeamsW, self.amp_scatters, self._update_amps, add_toolbar=False, set_lim=False,legend=False)
        self._create_polar_canvas(self.tab_3, self.polar_scatters, self._update_polar_canvas)

        self.tabWidget.setCurrentIndex(2)

        self._reset_color_buttons()

        for i in range(8):
            f = partial(self._clicked_button_general, i)
            getattr(self, f"stream_beam_{i + 2}").clicked.connect(f)

    def _reset_color_buttons(self):
        for i in range(2, 10):
            getattr(self, f"stream_beam_{i}").setStyleSheet("background-color : grey")

    def _clicked_button_general(self, b):
        self._reset_color_buttons()
        self.ac.ia_mic.current_beam_id = b
        getattr(self, f"stream_beam_{b + 2}").setStyleSheet("background-color : red")

    def _create_polar_canvas(self, widget, sc_l, updater):
        layout = QtWidgets.QVBoxLayout(widget)
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        _ax = dynamic_canvas.figure.subplots(subplot_kw={'projection': 'polar'})
        self.canvas.append(_ax)
        self.figures.append(dynamic_canvas)

        self.d.addToolBar(NavigationToolbar(dynamic_canvas, self.d))

        # creating initial data

        for m in self.ac.mic_api_names:
            sc = _ax.plot(0, 200, 'ro', label=m, alpha=1)
            sc_l.append(sc)

        # calling update function every 50 ms
        _ax.legend()
        timer = dynamic_canvas.new_timer(50)
        timer.add_callback(updater)
        timer.start()
        self.timers.append(timer)

    def _update_polar_canvas(self):
        _ax = self.canvas[3]
        _ax.cla()
        _ax.set_ylim(0, 100)

        data = self.ac.plotter_mic_combat.last_polar_data
        if not data:
            return


        k = data[0]

        trans = mtransforms.offset_copy(_ax.transData, fig=self.figures[0], y=6, units='dots')
        for x, y in zip(k.T[0], k.T[1]):
            _ax.plot(np.deg2rad(x), y, 'ro', label=f"{x}, {y}")
            _ax.plot(np.deg2rad(x), y, 'ro', label=f"{x}, {y}")
            # _ax.text(x, y, f"{round(x,1)}, {round(y,1)}", transform=trans,rotation=45)

        # plt.legend()

        # for sc in self.polar_scatters:
        self.polar_scatters[0][0].figure.canvas.draw()

    def _create_doa_canvas(self, widget, canvas, sc):
        pass

    def _create_ml_canvas(self, widget, canvas, sc):
        pass

    def _create_canvas(self, widget, sc_l, updater, add_toolbar=True, set_lim=True, ax_x=(0, 60), ax_y=(0, 360),
                       legend=True):
        layout = QtWidgets.QVBoxLayout(widget)
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        _ax = dynamic_canvas.figure.subplots()
        self.canvas.append(_ax)

        if add_toolbar:
            self.d.addToolBar(NavigationToolbar(dynamic_canvas, self.d))

        # creating initial data
        if set_lim:
            _ax.set_xlim(ax_x[0], ax_x[1])
            _ax.set_ylim(ax_y[0], ax_y[1])

        else:
            _ax.autoscale_view()

        for m in self.ac.mic_api_names:
            sc = _ax.scatter([], [], label=m, alpha=1)
            sc_l.append(sc)

        # calling update function every 50 ms
        if legend:
            _ax.legend()
        timer = dynamic_canvas.new_timer(1000)
        timer.add_callback(updater)
        timer.start()
        self.timers.append(timer)

    def _update_canvas_doa(self):
        return self._update_canvas_base(x=self.doa_scatters, y=self.ac.plotting_doa_data)

    def _update_amps(self):
        _ax = self.canvas[2]
        _ax.cla()
        _ax.imshow(self.ac.plotter_mic_aa.last_beams_amp, cmap='Reds')

        for i in range(4):
            value = self.rms_to_db(self.ac.plotter_mic_aa.last_beams_amp[0, i])
            _ax.text(i, 0, round(value, 1), rotation=45)

        for i in range(4):
            value = self.rms_to_db(self.ac.plotter_mic_aa.last_beams_amp[1, 3 - i])
            _ax.text(3 - i, 1, round(value, 1), rotation=45)

        for sc in self.amp_scatters:
            sc.figure.canvas.draw()

    def _update_canvas_ml(self):
        return self._update_canvas_base(x=self.ml_scatters, y=self.ac.plotting_ml_data, text_base=True,
                                        _ax=self.canvas[1])

    @staticmethod
    def _update_canvas_base(x, y, text_base=False, _ax=None):
        if text_base:

            _ax.cla()
            for sc, data in zip(x, y):
                _ax.plot(data[0], data[1], '*-')
                plt.yticks(rotation=90)
                _ax.legend()
                sc.figure.canvas.draw()
                # _ax.set_ylim(0,100)

            _ax.relim()
            _ax.autoscale()


        else:
            for sc, d in zip(x, y):
                sc.set_offsets(d)

                sc.figure.canvas.draw()

    def _reset_beam_values(self):
        beam_id = self.beam_io.value()
        beam, gain, phi, theta = self.ac.ia_mic.get_beam_data(beam_id)
        self.beam_io.setValue(int(beam))
        self.gain_io.setText(str(gain))
        self.phi_io.setValue(float(phi))
        self.theta_io.setValue(float(theta))

    def _set_values(self):
        beam = self.beam_io.value()
        gain = self.gain_io.text()
        phi = self.phi_io.value()
        theta = self.theta_io.value()
        self.ac.ia_mic.set_full_beam(str(beam), int(gain), float(phi), float(theta))

    @staticmethod
    def rms_to_db(x):
        return 20 * np.log(x)

    def _terminate(self):
        self.ac.terminate()
        sys.exit(0)


def main():
    acoustic_manager = EaServer(play_audio=False)
    acoustic_manager.connect_to_ba_mics()
    acoustic_manager.connect_to_ia_mics()
    print('----- connected to mics -----------')
    acoustic_manager.connect_to_dg()
    print('----- connected to drone gourd -----------')
    acoustic_manager.start_processing()
    print('----- processipipng data -----------')
    # acoustic_manager.send_results()
    print('----- sending data -----------')
    # acoustic_manager.write_to_file()
    acoustic_manager.write_results_log()
    # acoustic_manager.live_plot_doa()Error in opening stream on device

    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QMainWindow()
    ui = MainWindow(acoustic_manager)
    ui.setupUi(dialog)
    dialog.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
