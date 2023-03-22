import numpy as np


def celebamaskhq_colors():
    colormap = np.array([
        [0,  0,  0], 
        [204, 0,  0], # skin
        [76, 153, 0], # nose
        [204, 204, 0], # eye_g
        [51, 51, 255], # l_eye
        [204, 0, 204], # r_eye
        [0, 255, 255], # l_brow
        [255, 204, 204], # r_brow
        [102, 51, 0], # l_ear
        [255, 0, 0], # r_ear
        [102, 204, 0], # mouth
        [255, 255, 0], # u_lip
        [0, 0, 153], # l_lip
        [0, 0, 204], # hair
        [255, 51, 153], # hat
        [0, 204, 204], # ear_r
        [0, 51, 0], # neck_l
        [255, 153, 51], # neck
        [0, 204, 0]],  # cloth
        dtype=np.uint8
    ) 

    return colormap

def celebamaskhq_lite_colors():
    colormap = np.array([
        [0,  0,  0], 
        [204, 0,  0], # skin
        [76, 153, 0], # nose
        [204, 204, 0], # eye_g
        [51, 51, 255], # l_eye
        [0, 255, 255], # l_brow
        [102, 51, 0], # l_ear
        [102, 204, 0], # mouth
        [255, 255, 0], # u_lip
        [0, 0, 153], # l_lip
        [0, 0, 204], # hair
        [255, 51, 153], # hat
        [0, 204, 204], # ear_r
        [0, 51, 0], # neck_l
        [255, 153, 51], # neck
        [0, 204, 0]],  # cloth
        dtype=np.uint8
    ) 

    return colormap