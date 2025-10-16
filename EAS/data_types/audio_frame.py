# -*- coding: utf-8 -*-
"""

audio_frame.py - define a data structure to hold dji data with audio frame

written by: Rami. W

Change Log:
14-05-2020 : creation
"""

import enum
import sys
#import numpy

class Audio_frame:
    def __init__(self,position=(0, 0, 0),az = 0.0,rate = 48000,frame_list = []):
        self.position = position
        self.azimuth = az
        self.frame_rate = rate
        self.frame_list = frame_list

if __name__ == '__main__':
    fr = Audio_frame()
