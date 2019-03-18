'''
Test fitting of team member data with different classifiers
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import feature
from PIL import Image

''' Helpful commands:
        Display Image:
            plt.imshow(np.reshape(image_arr[i][j],(100,100,3)))
'''
 

def load_images():
    '''
    Load all images as RGB data in an array
    Classify all images
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letter_dirs = glob.glob('%s/TRM_Pics/Combined/*' % (script_dir))
    letter_dirs.sort()

    image_list = []
    image_class = []
    for curr_letter_dir in letter_dirs:
        curr_letter = curr_letter_dir[-1] # Get last element of string
        curr_letter = curr_letter.lower() # Make lowercase

        letter_image_list = []
        letter_image_class = []
        for filename in glob.glob('%s/*' % (curr_letter_dir)):
            im = Image.open(filename)
            imgRGB = list(im.getdata()) # a set of 3 values(R, G, B)
            letter_image_list.append(imgRGB) # Append RGB data list
            letter_image_class.append(curr_letter) # Append classification

        image_list.append(letter_image_list)
        image_class.append(letter_image_class)

    image_arr = np.asarray(image_list)
    image_class_arr = np.asarray(image_class)
    f = 1
    
    return (image_arr, image_class_arr, image_list, image_class)


if __name__ == "__main__":

    image_arr, image_class_arr, image_list, image_class = load_images()
    f = 1