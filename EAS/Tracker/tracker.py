from EAS.Tracker import Track, min_zero
from EAS.Tracker.intercept import Intercept


class Tracker:
    purge_time = 10
    min_hits_for_queue = 3
    prediction_distance_limit = 7
    max_score = 2
    no_match_score = 0.95

    def __init__(self):
        self.purge_time = Tracker.purge_time
        self.min_hits_for_queue = Tracker.min_hits_for_queue
        self.track_list = []
        self.purged_tracks = []
        self.prediction_distance_limit = Tracker.prediction_distance_limit

    def active_tracks(self, time):
        return filter(lambda x: time - x.last_intercept_time < self.purge_time, self.track_list)

    def add_frame_result(self, frame_result):
        inter = Intercept(frame_result.doaInDeg, frame_result.sensor, frame_result.classification,
                          1, frame_result.updateTimeTagInSec, frame_result)
        self.add_intercept(inter)

    def add_intercept(self, intercept):
        match_results = []

        for track in self.active_tracks(intercept.time):
            score = self._match_track(track, intercept)
            match_results.append(score)

        if not all(match_results) or not match_results:
            self._create_track(intercept)
            return

        max_score = max(match_results)
        [*self.active_tracks(intercept.time)][match_results.index(max_score)].add_intercept(intercept)

    def _match_track(self, track, intercept):
        score = 0

        if not self._match_sensor(track, intercept):
            return score

        if not self._match_track_time(track.last_intercept_time, intercept.time):
            return score

        angle_calc = track.predict(intercept)
        score += self._match_geo(angle_calc, intercept)

        if score == 0:
            return score

        ml_score = self._match_ml_class(track, intercept)
        score = 0 if ml_score < 0 else score + ml_score
        return score

    @staticmethod
    def _match_sensor(track, intercept):
        return track.sensor == intercept.sensor

    @staticmethod
    def _match_ml_class(track, intercept):
        if track.main_identification != 'unknown' and track.main_identification != intercept.ml_class and intercept.ml_score > Tracker.no_match_score:
            return -1
        return track.main_identification == intercept.ml_class and track.main_identification == intercept.ml_class != 'unknown'

    def _match_geo(self, angle_calc, intercept):
        # fixing 360 fold
        angle_calc = angle_calc - 360 if angle_calc > 360 else angle_calc

        # fixing fold
        if 360 - self.prediction_distance_limit >= (angle_calc - intercept.doa) >= self.prediction_distance_limit:
            return False

        # fixing fold
        if self.prediction_distance_limit - 360 <= (angle_calc - intercept.doa) <= -self.prediction_distance_limit:
            return False

        return round(1 - min_zero(intercept.doa - angle_calc, (intercept.doa - angle_calc) % 360), 4)

    def _match_track_time(self, last_intercept_time, intercept_time):
        if intercept_time - last_intercept_time > self.purge_time:
            return False
        return True

    def _create_track(self, intercept):
        self.track_list.append(Track(intercept))
