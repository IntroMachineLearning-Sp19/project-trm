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

from scipy import ndimage as ndi
from skimage import feature

from sklearn import preprocessing as preprocess
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import cv2

from tqdm import tqdm

def train_score_image_classifier():
    rgb_image_arr, hsv_image_arr, image_class_arr = load_images()

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


def load_images():
    '''
    Load all images as RGB data in an array
    Classify all images
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letter_dirs = glob.glob('%s/TRM_Pics/Combined/*' % (script_dir))
    letter_dirs.sort()

    rgb_image_list = []
    hsv_image_list = []
    image_class = []
    for curr_letter_dir in letter_dirs:
        curr_letter = curr_letter_dir[-1] # Get last element of string
        curr_letter = curr_letter.lower() # Make lowercase

        for filename in glob.glob('%s/*' % (curr_letter_dir)):
            im = Image.open(filename)

            img_rgb = list(im.getdata()) # a set of 3 values(R, G, B)
            rgb_image_list.append(img_rgb) # Append RGB data list

            img_hsv = list(im.convert('HSV').getdata())
            hsv_image_list.append(img_hsv) # Append HSV data list
            image_class.append(curr_letter) # Append classification

    # Convert lists to arrays
    rgb_image_arr = np.asarray(rgb_image_list, dtype=np.uint8)
    hsv_image_arr = np.asarray(hsv_image_list, dtype=np.uint8)
    image_class_arr = np.asarray(image_class) # TODO: Convert chars to ASCII vals

    # Convert data arrays to [(num_images)x100x100x3]
    num_images = len(rgb_image_arr)
    rgb_image_arr = np.reshape(rgb_image_arr, (num_images, 100, 100, 3))
    hsv_image_arr = np.reshape(hsv_image_arr, (num_images, 100, 100, 3))

    return (rgb_image_arr, hsv_image_arr, image_class_arr)

def preprocess_image(rgb_preimg, hsv_preimg):
    '''
    Preprocess an image for feature extraction and classification
    '''

    # Detect hand and crop edges to tightly enclose it
    img_saturation = hsv_preimg[:, :, 1]

    crop_img = hsv_preimg[45:55, 45:55]

    average_hand = np.mean(crop_img, dtype=np.float64)

    h_mask = (hsv_preimg >= average_hand) & (hsv_preimg <= 255)

    # DEBUG
    # plt.imshow(h_mask)
    # plt.show()

    center_x = 50
    center_y = 50
    window_width = 5

    row_sum1 = 0
    row_sum2 = 0

    col_sum1 = 0
    col_sum2 = 0

    crop_img = h_mask[center_x-window_width:center_x+window_width,
                      center_y-window_width:center_y+window_width]

    while 1:
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
            
        window_width += 5
        crop_img = h_mask[center_x-window_width:center_y+window_width, center_x-window_width:center_y+window_width]
        
        # DEBUG
        # print(average)
        # print(average_com/average_hand)
        if average_com/average_hand > 3.0:
            break
        elif window_width == 45:
            break
        else: 
            average     = 0
            row_sum1    = 0
            row_sum2    = 0
            col_sum1    = 0
            col_sum2    = 0


    crop_img = hsv_preimg[center_x-window_width:center_x+window_width,
                          center_y-window_width:center_y+window_width]

    # Scale up to 100x100 if needed
    resized_img = cv2.resize(crop_img, (100, 100))

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

def knn_classifier(data_arr, class_arr, k_max=10):
    '''
    Fit and test a KNN classifier on input data
    data_arr should be [N x M] where N is data points and M is features
    class_arr should be [N x 1]
    k_max: Maximum number to iterate up to in knn
    '''

    start_time = time.time()

    uniform_scores = np.zeros(k_max, dtype=np.float64)
    weighted_scores = np.zeros(k_max, dtype=np.float64)

    # Set up the classifiers we will run
    # K-NN with uniform distance and with weighted distances
    for i in tqdm(range(k_max)): # exclusive of the 5

        # X = data_arr.reshape(len(data_arr),30000) #feature dataset
        # Y = class_arr.reshape(len(data_arr)) #ground truth

        n_neighbors = i+1

        classifiers = []
        classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
        classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
        names = ['K-NN_Uniform', 'K-NN_Weighted']
        X_train, X_test, y_train, y_test = train_test_split(data_arr, class_arr, test_size=.1)


        # Iterate over classifiers
        for name, clf in zip(names, classifiers): 

            #Train the classifier
            clf.fit(X_train, y_train)

            #Test the classifier
            if name == 'K-NN_Uniform':
                uniform_scores[i] = (clf.score(X_test, y_test))
            else:
                weighted_scores[i] = (clf.score(X_test, y_test))

    plt.figure()
    plt.scatter(x=np.arange(start=1, stop=k_max+1), y=uniform_scores)
    plt.title("K from 1 to {}: Uniformed K-NN".format(k_max))
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")

    plt.figure()
    plt.scatter(x=np.arange(start=1, stop=k_max+1), y=weighted_scores)
    plt.title("K from 1 to {}: Weighted K-NN".format(k_max))
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    plt.show()

    end_time = time.time()
    print("Knn Runtime: {} seconds".format(end_time - start_time))

    f = 1

def random_forest_classifier(data_arr, class_arr, max_trees=50):
    '''
    Fit and test a random forest classifier on input data
    data_arr should be [N x M] where N is data points and M is features
    class_arr should be [N x 1]
    Max Trees must be a multiple of 10
    '''
    num_trees = math.floor(max_trees/10)
    start_time = time.time()
    num_folds = 10
    num_features = 10
    max_depth = 20
    avg_indx = 0
    std_indx = 1

    class FoldClass:
        avg = 0
        std = 0

    class DataClass:
        rawScores = []
        foldScores = []
        def __init__(self):
            for i in range(num_folds):
                self.foldScores.append(FoldClass)

    class RandForestClass():
        treeNum = []
        treeDepth = []
        dataFeatures = []
        def __init__(self, numTrees=num_trees):
            for i in range(numTrees):
                self.treeNum.append(DataClass())
            for i in range(max_depth):
                self.treeDepth.append(DataClass())
            for i in range(num_features+1):
                self.dataFeatures.append(DataClass())

    # paviaSpectra = data.reshape(len(data),30000)
    # gtList = ground_truth.reshape(len(data))

    # treeScores -> x0 = number of trees, x1 = test fold, x2 = runs
    treeScores = RandForestClass()
    plot_x = np.zeros(num_trees)
    plot_y = np.zeros((num_folds, 2, num_trees))

    for i in tqdm(range(num_trees)):
        # Define classifier/estimator
        if i == 0:
            num_estimators = i+1
        else:
            num_estimators = (i)*10

        forest = RandomForestClassifier(criterion='entropy',
                                        n_estimators=num_estimators, n_jobs=-1)

        runs = []
        #print("Number of Trees: ", num_estimators)
        for j in range(3):
            runs.append(cross_val_score(forest, data_arr, y=class_arr,
                                        cv=num_folds, n_jobs=-1))

        # Makes the test folds row mapped and run_number column mapped
        # test_fold 1 -> runs[0][0], runs[0][1], runs[0][2] -> runs[0][:]
        treeScores.treeNum[i].rawScores = np.copy((np.column_stack(runs)))

        #print(treeScores.treeNum[i].rawScores)

        for fold in range(num_folds):
            treeScores.treeNum[i].foldScores[fold].avg = \
                np.copy(np.mean(treeScores.treeNum[i].rawScores[fold]))
            treeScores.treeNum[i].foldScores[fold].std = \
                np.copy(np.std(treeScores.treeNum[i].rawScores[fold]))
            plot_y[fold][avg_indx][i] = \
                np.copy(treeScores.treeNum[i].foldScores[fold].avg)
            plot_y[fold][std_indx][i] = \
                np.copy(treeScores.treeNum[i].foldScores[fold].std)
            #print(plot_y[fold][avg_indx][i])


    for i in range(num_trees):
        if i == 0:
            num_estimators = i+1
        else:
            num_estimators = (i)*10
        plot_x[i] = np.copy(num_estimators)

    plt.figure(figsize=(20, 10))
    plt.figure(1)

    # Multiple line plot
    for fold_num in range(num_folds):
        # print(plot_y[fold_num][avg_indx][:])
        plt.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:],
                     label='Test Fold {0}'.format(fold_num))    

    plt.xlabel('Number of Trees', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Pavia Dataset Random Trees Classification', fontsize=20)
    plt.legend()
    plt.show()

    end_time = time.time()
    print("Random Forest Runtime: {} seconds".format(end_time - start_time))

    f = 1

if __name__ == "__main__":
    # # Run this code only on ubuntu
    # if os.name == "posix":
    #     # Change the method with which new threads are created, required for sklearn multithreading
    #     # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    #     import multiprocessing
    #     multiprocessing.set_start_method('forkserver')

    # rgb_image_arr, hsv_image_arr, image_class_arr = load_images()
    rgb_image_arr, hsv_image_arr = load_image_single('/home/michael/Documents/EEL4930-ML/Project/2019_sp_ml_train_data/c/C_21.JPG')
    extracted_features = extract_features(rgb_image_arr)
    
    f = 1

''' Helpful commands:
        Display Image:
            plt.imshow(np.reshape(image_arr[i][j],(100,100,3)))
'''
