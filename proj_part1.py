'''
Test fitting of team member data with different classifiers
'''

import os
import numpy as np
import glob as glob
from PIL import Image

def load_images():
    '''
    Load all images as RGB data in an array
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letter_dirs = glob.glob('%s/TRM_Pics/Combined/*' % (script_dir))
    letter_dirs.sort()

    image_list = []
    for curr_letter_dir in letter_dirs:
        letter_image_list = []
        for filename in glob.glob('%s/*' % (curr_letter_dir)):
            im = Image.open(filename)
            imgRGB = list(im.getdata())    # a set of 4 values(R, G, B, A)
            letter_image_list.append(imgRGB)
        image_list.append(letter_image_list)

    f = 1




if __name__ == "__main__":

    load_images()

    f = 1
