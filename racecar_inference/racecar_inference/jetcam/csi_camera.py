import atexit
import threading

import cv2
import numpy as np
import traitlets

from .camera import Camera


class CSICamera(Camera):

    capture_device = traitlets.Integer(default_value=0)
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=1280)
    capture_height = traitlets.Integer(default_value=720)

    # crop in the 1280x720 pic
    # TODO config this from yaml files
    CROP_X, CROP_Y, CROP_W, CROP_H = 698, 331, 750, 500
    CROP_LEFT = int(CROP_X - CROP_W / 2)
    CROP_TOP = int(CROP_Y - CROP_H / 2)
    CROP_RIGHT = int(CROP_X + CROP_W / 2)
    CROP_BOTTOM = int(CROP_Y + CROP_H / 2)

    def __init__(self, *args, **kwargs):
        super(CSICamera, self).__init__(*args, **kwargs)
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()

            if not re:
                raise RuntimeError("Could not read image from camera.")
        except:
            raise RuntimeError("Could not initialize camera.  Please see error trace.")

        atexit.register(self.cap.release)

    def _gst_str(self):
        # gst string if image is too bright
        # return 'nvarguscamerasrc sensor-id=%d exposurecompensation="-2" gainrange="10 10" aelock=true exposuretimerange="34000 34001" ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
        #        self.capture_device, self.capture_width, self.capture_height, self.capture_fps, self.width, self.height)

        # default gst string
        # return 'nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
        #         self.capture_device, self.capture_width, self.capture_height, self.capture_fps, self.width, self.height)

        return (
            f"nvarguscamerasrc sensor-id={self.capture_device} ! video/x-raw(memory:NVMM), width={self.capture_width}, height={self.capture_height}, "
            f"format=(string)NV12, framerate=(fraction){self.capture_fps}/1"
            f" ! nvvidconv top={self.CROP_TOP} bottom={self.CROP_BOTTOM} left={self.CROP_LEFT} right={self.CROP_RIGHT} ! video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            "format=(string)BGRx ! videoconvert ! appsink"
        )

    def _read(self):
        re, image = self.cap.read()
        if re:
            return image
        else:
            raise RuntimeError("Could not read image from camera")
