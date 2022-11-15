from enum import Enum


class VisionClasses(Enum):
    BACKGROUND = 0
    Mars = 1
    Snickers = 2
    Milky = 3


STREAM_RESOLUTION = (3264, 2464)
CONVERTER_RESOLUTION = (1280, 720)
NETWORK_RESOLUTION = (300, 300)
DIGITAL_GAIN_RANGE = (1, 1)
GAIN_RANGE = (3, 3)
AWB_MODE = 8
WB_GAINS = (200, 200, 200, 200)
DETECTION_THRESH = 0.6
NON_MAX_THRESH = 0.5
IMG_MEAN = 127.5
IMG_STD = 0.007843


LINE_THICKNESS = 1
CIRCLE_SIZE = 10
CIRCLE_FILL = (255, 0, 0)
