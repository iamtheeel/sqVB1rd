#
# From MIC
#

import cv2
import numpy as np
class RGB2RGB565(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        numpy_array = np.asarray(sample)
        #numpy_array = (numpy_array >> 8) + (numpy_array << 8) # Little to big endian, or visa/versa
        #image = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2BGR565) # convert BRG to RGB565  CAM_IMAGE_PIX_FMT_RGB565
        image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR565) # convert RGB to RGB565  CAM_IMAGE_PIX_FMT_RGB565, MIC, this is corect.
        return image