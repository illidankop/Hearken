# -*- coding: utf-8 -*-
"""

dg_frame.py - define a data structure to hold result of processing an audio frame

written by: Erez. S

Change Log:
05-02-2020 : creation
"""

import enum
import sys


# Using enum class create enumerations
class MovInd(str, enum.Enum):
    UNKONWN = 'UNKONWN'
    INCOMING = 'INCOMING'
    OUTGOING = 'OUTGOING'


class RangeInd(str, enum.Enum):
    UNKONWN = "UNKONWN"
    CLOSE = "CLOSE"
    MED = 'MED'
    FAR = 'FAR'


class DroneClass(str, enum.Enum):
    UNKONWN = 'UNKONWN'
    DJI4 = 'DJI4'
    INSPIRE = 'INSPIRE'
    MEMBO = 'MEMB'
    BEEBOP = 'BEEBOP'

class FrameResult:
    def __init__(self, unitId="Mic_0", msgCount=0, updateTimeTagInSec=0, hasDetection=True, movmentIndication=MovInd.OUTGOING, \
                rangeIndication=RangeInd.UNKONWN, droneClass=DroneClass.UNKONWN,class_confidence=None, doaInDeg=360.0 \
                ,doaInDegErr=1.1, elevationInDeg=90.0,elevationInDegErr=2.2,snr=0.0,detection=None):
        self.unitId = unitId  # Mic Id
        self.updateTimeTagInSec = updateTimeTagInSec
        self.movmentIndication = movmentIndication
        self.rangeIndication = rangeIndication
        self.classification = droneClass
        self.class_confidence = class_confidence
        self.doaInDeg = doaInDeg
        self.doaInDegErr = doaInDegErr
        self.elevationInDeg = elevationInDeg
        self.elevationInDegErr = elevationInDegErr
        self.snr = snr
        self.detection=detection        

