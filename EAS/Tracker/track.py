from collections import Counter


def min_zero(a, b):
    return b if abs(a) >= abs(b) >= 0 >= -1 * abs(b) >= -1 * abs(a) else a


class Track:
    alpha = 0.85
    beta = 0.1
    w = 0.2
    identification_score = 0.7

    def __init__(self, intercept):
        self.sensor = intercept.sensor
        self.alpha = Track.alpha
        self.beta = Track.beta
        self.w = Track.w
        self.intercepts_org = [intercept]
        self.intercepts_calc = [intercept]
        self.last_sent_time = 0

    def predict(self, intercept):
        dt = intercept.time - self.last_intercept_time
        angle_calc = self.last_intercept_angle + self.w * dt
        angle_error = min_zero((intercept.doa - angle_calc), (intercept.doa - angle_calc) % 360)
        angle_calc = angle_calc + self.alpha * angle_error
        return angle_calc

    def add_intercept(self, intercept):
        self._update_values(intercept)

    def _update_values(self, intercept):
        dt = intercept.time - self.last_intercept_time
        angle_calc = self.last_intercept_angle + self.w * dt
        angle_error = min_zero((intercept.doa - angle_calc), (intercept.doa - angle_calc) % 360)
        angle_calc = angle_calc + self.alpha * angle_error
        self.w = self.w + (self.beta / dt) * angle_error
        self.intercepts_org.append(intercept)
        self.intercepts_calc.append(intercept.change_doa(angle_calc))  # creating a new intercept object

    @property
    def main_identification(self):
        c = Counter([intercept.ml_class for intercept in self.intercepts_org])
        if not c:
            return 'unknown'

        else:
            k, v = c.most_common(1)[0]

        if v < 10:
            return 'unknown'

        elif v / len(self.intercepts_org) < Track.identification_score:
            return 'unknown'

        return k

    @property
    def last_intercept(self):
        return self.intercepts_calc[-1]

    @property
    def last_intercept_time(self):
        return self.intercepts_org[-1].time

    @property
    def last_intercept_angle(self):
        return self.intercepts_calc[-1].doa

    @property
    def times(self):
        return [x.time for x in self.intercepts_calc]

    @property
    def doas(self):
        return [x.doa for x in self.intercepts_calc]

    def __len__(self):
        return len(self.intercepts_calc)
