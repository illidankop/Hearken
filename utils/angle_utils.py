import numpy as np

class AngleUtils:

    @staticmethod
    def rad2deg(angle):
        return angle * 180 / np.pi

    @staticmethod
    def deg2rad(angle):
        return angle * np.pi / 180

    @staticmethod
    def is_angle_in_ranges(angle, angle_ranges):
        return any(start <= angle <= end for start, end in angle_ranges)
    
    @staticmethod
    def norm_rad(angle):
        norm_angle = angle
        if angle < 0:
            norm_angle += np.pi
        elif angle > np.pi:
            norm_angle -= np.pi
        return norm_angle

    @staticmethod
    def norm_deg(angle):
        norm_angle = angle
        if angle < 0:
            norm_angle += 360
        elif angle > 360:
            norm_angle -= 360
        return norm_angle
