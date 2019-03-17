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
            letter_image_list.append(im)
        image_list.append(letter_image_list)

    f = 1




if __name__ == "__main__":
    # Change the method with which new threads are created, required for sklearn multithreading
    # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    import multiprocessing
    multiprocessing.set_start_method('forkserver')

    load_images()

    f = 1
