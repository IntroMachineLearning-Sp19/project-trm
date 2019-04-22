'''
Test fitting of team member data with different classifiers
'''

import os
import glob
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter

from scipy import ndimage as ndi
from skimage import feature

from sklearn import preprocessing as preprocess
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import cv2

from tqdm import tqdm

import pickle

def load_image_single(filename):
    '''
    Load and return a single image as a 100x100x3 array
    '''
    # Open Image
    im = Image.open(filename)

    img_rgb = list(im.getdata()) # a set of 3 values(R, G, B)

    img_hsv = list(im.convert('HSV').getdata())

    # Convert lists to arrays
    rgb_image_arr = np.asarray(img_rgb, dtype=np.uint8)
    hsv_image_arr = np.asarray(img_hsv, dtype=np.uint8)

    # Convert data arrays to [(num_images)x100x100x3]
    rgb_image_arr = np.reshape(rgb_image_arr, (100, 100, 3))
    hsv_image_arr = np.reshape(hsv_image_arr, (100, 100, 3))

    return (rgb_image_arr, hsv_image_arr)


def load_images(path):
    '''
    Load all images as RGB data in an array
    Classify all images
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letter_dirs = glob.glob('{}/TRM_Pics/{}/*'.format(script_dir, path))
    letter_dirs.sort()

    rgb_image_list = []
    hsv_image_list = []
    image_class = []
    for curr_letter_dir in letter_dirs:
        curr_letter = curr_letter_dir[-1] # Get last element of string
        curr_letter = curr_letter.lower() # Make lowercase

        for filename in glob.glob('%s/*' % (curr_letter_dir)):
            im = Image.open(filename)
            
            im = im.resize((100, 100), Image.NEAREST)
            im = im.filter(ImageFilter.BoxBlur(3))

            img_rgb = list(im.getdata()) # a set of 3 values(R, G, B)
            rgb_image_list.append(img_rgb) # Append RGB data list

            img_hsv = list(im.convert('HSV').getdata())
            hsv_image_list.append(img_hsv) # Append HSV data list
            image_class.append(curr_letter) # Append classification

    # Convert lists to arrays
    rgb_image_arr = (np.asarray(rgb_image_list, dtype=np.uint8))
    hsv_image_arr = np.asarray(hsv_image_list, dtype=np.uint8)
    image_class_arr = np.asarray(image_class) # TODO: Convert chars to ASCII vals
    
    rgb_image_arr = rgb_image_arr.reshape(len(rgb_image_arr),100,100,3)
    hsv_image_arr = hsv_image_arr.reshape(len(rgb_image_arr),100,100,3)

    # Convert data arrays to [(num_images)x100x100x3]
    num_images = len(rgb_image_arr)
    rgb_image_arr = np.reshape(rgb_image_arr, (num_images, 100, 100, 3))
    hsv_image_arr = np.reshape(hsv_image_arr, (num_images, 100, 100, 3))

    return (rgb_image_arr, hsv_image_arr, image_class_arr)

def preprocess_each_image(hsv):
    '''
    Preprocess an image for feature extraction and classification
    '''

    # Detect hand and crop edges to tightly enclose it
    im_hue = hsv[:,:,1]
    
    crop_img = im_hue[45:55, 45:55]
    
    average_hand = np.mean(crop_img, dtype=np.float64)
    
    max_hand = np.max(crop_img)
    low_hand = np.min(crop_img)

    h_mask = (im_hue >= low_hand) & (im_hue <= max_hand)
    
#    h_mask = (im_hue >= average_hand) & (im_hue <= 255)
    
    #imgplot = plt.imshow(h_mask)
    #plt.show()
    
    x = 45
    y = 55
    h = 0
    
    row_sum1 = 0
    row_sum2 = 0
    
    col_sum1 = 0
    col_sum2 = 0
    
    crop_img = h_mask[x-h:y+h, x-h:y+h]
    
    
    while(1):
        i = 0
        for i in range( len(crop_img) - 1):
            if (crop_img[0, i] == False):
                row_sum1 += 1
            elif (crop_img[len(crop_img)-1, i] == False):
                row_sum2 += 1
            elif (crop_img[i, 0] == False):
                col_sum1 += 1
            elif (crop_img[i, len(crop_img)-1] == False):
                col_sum2 += 1
            
            i+=1
        
        average = (row_sum1 + row_sum2 + col_sum1 + col_sum2) / (4 * (i-1))
        average_com = (4 * (i-1))
            
        h += 5
        crop_img = h_mask[x-h:y+h, x-h:y+h]
        
        #print(average)
        #print(average_com/average_hand)
        if average_com/average_hand >= 1.5:
            break
        elif h == 45:
            break
        else: 
            average     = 0
            row_sum1    = 0
            row_sum2    = 0
            col_sum1    = 0
            col_sum2    = 0
    
    
    crop_img = im_hue[x-h:y+h, x-h:y+h]
    
    resize_img = cv2.resize(crop_img,(100,100))
    return resize_img

    # TODO: Delete background


    f = 1

def extract_features(image, kernel_size=3):
    '''
    Extract new features from the image for use in classification
    New features: edge histogram, ORB keypoints
    '''

    extracted_features = []

    # Gaussian blur image
    blur_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Calculate the type (horizontal, diagonal, vertical) and location of edges
    # perform Sobel edge detection in x and y dimensions
    edges_x = cv2.Sobel(blur_img, -1, 1, 0)
    edges_y = cv2.Sobel(blur_img, -1, 0, 1)

    # Partition and average x and y edges in 9 equal sections of the image
    edge_avg = np.zeros((2, 9))
    i = 0
    cutoffs = [(0, 33), (33, 66), (66, 100)]
    for lower_x_cutoff, upper_x_cutoff in cutoffs:
        for lower_y_cutoff, upper_y_cutoff in cutoffs:
            edge_avg[0, i] = np.mean(edges_x[lower_x_cutoff:upper_x_cutoff, lower_y_cutoff:upper_y_cutoff])
            edge_avg[1, i] = np.mean(edges_y[lower_x_cutoff:upper_x_cutoff, lower_y_cutoff:upper_y_cutoff])
            i = i + 1


    # DEBUG, show progression of 
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(blur_img)
    # plt.show()
    # plt.imshow(edges_x)
    # plt.show()
    # plt.imshow(edges_y)
    # plt.show()

    extracted_features.append(edge_avg.flatten())

    return np.asarray(extracted_features) # Return array of concatenated extracted features

    f = 1
    
def preprocess_all_images(hsv_image_arr_input):
    hsv_image_arr_pp = [];
    for i in range(len(hsv_image_arr_input)):
        hsv_image_arr_pp.append(preprocess_each_image(hsv_image_arr_input[i]))
    return np.asarray(hsv_image_arr_pp)

def extract_features_all_images(hsv_image_arr_input):
    hsv_image_arr_ef = [];
    for i in range(len(hsv_image_arr_input)):
        hsv_image_arr_ef.append(extract_features(hsv_image_arr_input[i]))
    return np.asarray(hsv_image_arr_ef).squeeze()

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
    
def test(testData):
    with open('randomForestModel.pk1', 'rb') as f:
        model = pickle.load(f)
    
    hsv_image_arr_af = testData
    
    hsv_image_arr_af = preprocess_all_images(hsv_image_arr_af)
    hsv_image_arr_feats_af = extract_features_all_images(hsv_image_arr_af)
    hsv_image_arr_af = hsv_image_arr_af.reshape(len(hsv_image_arr_af),10000)
    hsv_image_arr_af = np.concatenate((hsv_image_arr_af, hsv_image_arr_feats_af),axis=1)       
    
#    hsv_image_arr_af, image_class_arr_af = shuffle_in_unison(hsv_image_arr_af, image_class_arr_af)    Need image labels to get score. Could add as another parameter
#    print(paths[1], 'score: ', model.score(hsv_image_arr_af, image_class_arr_af))
    af_prediction_labels = (model.predict(hsv_image_arr_af)).tolist()

    with open("estimatedLabels.txt", "w") as file:
        file.write(str(af_prediction_labels))

if __name__ == "__main__":
        
    plt.show()
    
    paths = ["2019_sp_ml_train_data", "A_n_F", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
    easyRGB, easyHSV, easyLabels = load_images(paths[1])
    
    paths = ["2019_sp_ml_train_data", "A_n_F", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
    hardRGB, hardHSV, hardLabels = load_images(paths[0])
    
    with open("easyLabels.txt", "w") as file:
        file.write(str(easyLabels.tolist()))
        
    with open("hardLabels.txt", "w") as file:
        file.write(str(hardLabels.tolist()))

    test(easyRGB)

''' Helpful commands:
        Display Image:
            plt.imshow(np.reshape(image_arr[i][j],(100,100,3)))
'''
