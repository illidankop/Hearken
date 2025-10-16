import copy


class Intercept:
    def __init__(self, doa, sensor, ml_class, ml_score, time, frame_result=None):
        self.doa = doa
        self.sensor = sensor
        self.ml_class = ml_class
        self.ml_score = ml_score
        self.time = time
        self.frame_result = frame_result

    def change_doa(self, doa):
        new = copy.deepcopy(self)
        new.doa = doa
        return new
