#!/usr/bin/env python
import platform
import cv2
from pylab import *
import matplotlib.pyplot as plt
import numpy as np


"""Filters"""

def apply_filters(frame, flrs):

    brt = False
    cns = False
    zom = False
    rot = False
    fl_frame = None

    if flrs[0] != 0: #Brightness
        br_frame = cv2.add(np.array([flrs[0]]), frame)
        brt = True

        if flrs[1] != 1: # Contrast
            cn_frame = cv2.multiply(br_frame, np.array([flrs[1]]))
            cns = True

    else :
        if flrs[1] != 1:  # Contrast
            cn_frame = cv2.multiply(frame, np.array([flrs[1]]))
            cns = True



    if flrs[2] != 1 or flrs[3] != 0: # Zooming & Rotation
        rows, cols = frame.shape[:2]

        if flrs[2] != 1:
            zom = True
            if cns:
                zm_frame = zooming(cn_frame, flrs[2], rows, cols)

            else :
                if brt:
                    zm_frame = zooming(br_frame, flrs[2], rows, cols)

                else :
                    zm_frame = zooming(frame, flrs[2], rows, cols)


        if flrs[3] != 0 :
            rot = True
            root = cv2.getRotationMatrix2D((cols / 2, rows / 2), flrs[3], 1)
            if zom :
                rt_frame = cv2.warpAffine(zm_frame, root, (cols, rows))

            else :
                if cns :
                    rt_frame = cv2.warpAffine(cn_frame, root, (cols, rows))

                else:
                    if brt:
                        rt_frame = cv2.warpAffine(br_frame, root, (cols, rows))

                    else:
                        rt_frame = cv2.warpAffine(frame, root, (cols, rows))

    if rot:
        fl_frame = rt_frame

    else:
        if zom:
            fl_frame = zm_frame
        else:
            if cns:
                fl_frame = cn_frame
            else:
                if brt:
                    fl_frame = br_frame
                else:
                    fl_frame = frame


    return fl_frame







def zooming(img, sc, old_height, old_width):
    o = None
    new_height, new_width = int(sc * old_height), int(sc * old_width)
    if sc > 0:
        width_start = (new_width / 2) - (old_width / 2)
        width_end = (new_width / 2) + (old_width / 2)
        height_start = (new_height / 2) - (old_height / 2)
        height_end = (new_height / 2) + (old_height / 2)
        i = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        o = i[height_start:height_end+1, width_start:width_end+1]

    return o


