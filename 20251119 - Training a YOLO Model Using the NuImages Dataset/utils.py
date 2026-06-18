import numpy as np

def PxyXY_to_Nxcycwh(xyXY, width_pixels, height_pixels):

    # convert input to numpy array
    xyXY = np.asarray(xyXY, dtype=np.float32)

    # calculate width and height of bounding box
    box_width_height = xyXY[2:] - xyXY[:2]

    # calculate center of bounding box
    box_center = xyXY[:2] + box_width_height / 2.0

    # normalize coordinates to [0, 1] range
    normalized = np.hstack((box_center / [width_pixels, height_pixels],
                            box_width_height / [width_pixels, height_pixels]))
    
    return normalized