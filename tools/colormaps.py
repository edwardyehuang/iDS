import numpy as np


def celebamaskhq_colors():
    colormap = [
        [0, 0, 0], # background
        [255, 85, 0], # skin 
        [255, 0, 85], # l_brow
        [255, 0, 85], # r_brow
        [0, 0, 255], # l_eye
        [0, 0, 255], # r_eye
        [85, 255, 0], # eye_g
        [0, 170, 255], # l_ear
        [0, 170, 255], # r_ear
        [0, 255, 170], # ear_r
        [0, 255, 0], # nose
        [85, 0, 255], # mouth
        [170, 0, 255], # u_lip
        [0, 85, 255], # l_lip
        [255, 170, 0], # neck
        [255, 170, 0], # neck_l
        [255, 255, 85], # cloth
        [255, 0, 0], # hair
        [255, 0, 255] # hat
    ]

    return colormap



def celebamaskhq_lite_colors():
    colormap = [
        [0, 0, 0], # background
        [255, 85, 0], # skin 
        [255, 0, 85], # l_brow
        [0, 0, 255], # l_eye
        [85, 255, 0], # eye_g
        [0, 170, 255], # l_ear
        [0, 255, 170], # ear_r
        [0, 255, 0], # nose
        [85, 0, 255], # mouth
        [170, 0, 255], # u_lip
        [255, 170, 0], # neck
        [255, 170, 0], # neck_l
        [255, 255, 85], # cloth
        [255, 0, 0], # hair
        [255, 0, 255] # hat
    ]

    colormap = np.array(colormap)

    return colormap