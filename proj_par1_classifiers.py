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
import cv2

import numpy as np
import time
import math
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors, datasets
from sklearn import preprocessing as preprocess
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

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

        for filename in glob.glob('%s/*' % (curr_letter_dir)):
            im = Image.open(filename)
            imgRGB = list(im.getdata()) # a set of 3 values(R, G, B)
            image_list.append(imgRGB) # Append RGB data list
            image_class.append(curr_letter) # Append classification

    image_arr = np.asarray(image_list)
    image_class_arr = np.asarray(image_class)
    
#    image_arr = image_arr.reshape(len(image_arr),30000);
#    image_class_arr = image_class_arr.reshape(len(image_class_arr)) #ground truth
    f = 1
    
    return (image_arr, image_class_arr, image_list, image_class)

def extract_orb_features(image):
    img = cv2.imread(image,0)

    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=50)

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    kp1 = kp[0]

    kp_array = []
    for (i,keypoint) in enumerate(kp):
        kp_array.append(keypoint.angle)
        kp_array.append(keypoint.octave)
        kp_array.append(keypoint.pt[0])
        kp_array.append(keypoint.pt[1])
        kp_array.append(keypoint.response)
        kp_array.append(keypoint.size)
        img2 = img.copy()
    for marker in kp:
        img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

    plt.imshow(img2),plt.show()

def randfom_forest_test(data, ground_truth, max_trees):
    '''Max Trees must be a multiple of 10'''
    num_trees = math.floor(max_trees/10)+1
    start = time.time()
    num_folds = 10
    num_features = 10
    maxDepth = 20
    avg_indx = 0
    std_indx = 1
    
    class foldClass:
        avg = 0
        std = 0

    class dataClass:
        rawScores =  []
        foldScores = []
        def __init__(self):
            for i in range(num_folds):
                self.foldScores.append(foldClass)
        
    class randForestClass():
        treeNum = []
        treeDepth = []
        dataFeatures = []
        def __init__(self, numTrees=num_trees):
            for i in range(numTrees):
                self.treeNum.append(dataClass())
            for i in range(maxDepth):
                self.treeDepth.append(dataClass())
            for i in range(num_features+1):
                self.dataFeatures.append(dataClass())
    
    paviaSpectra = data
    gtList = ground_truth
            
    treeScores = randForestClass();    #treeScores -> x0 = number of trees, x1 = test fold, x2 = runs
    plot_x = np.zeros(num_trees)
    plot_y = np.zeros((num_folds, 2, num_trees))
    
    for i in tqdm(range(num_trees)):
        #define classifier/estimator
        if i == 0:
            num_estimators = i+1
        else:
            num_estimators = (i)*10
            
        forest = RandomForestClassifier(criterion='entropy', n_estimators=num_estimators, n_jobs=-1)
        
        runs = []
        #print("Number of Trees: ", num_estimators)
        for j in range(3):
            runs.append(cross_val_score(forest, paviaSpectra, y=gtList, cv=num_folds, n_jobs=-1))
        treeScores.treeNum[i].rawScores = np.copy((np.column_stack(runs))) #makes the test folds row mapped and run_number column mapped
                                                 #so test_fold 1 -> runs[0][0], runs[0][1], runs[0][2] -> runs[0][:]
        #print(treeScores.treeNum[i].rawScores)
        
        for fold in range(num_folds):
            treeScores.treeNum[i].foldScores[fold].avg = np.copy(np.mean(treeScores.treeNum[i].rawScores[fold]))
            treeScores.treeNum[i].foldScores[fold].std = np.copy(np.std(treeScores.treeNum[i].rawScores[fold]))
            plot_y[fold][avg_indx][i] = np.copy(treeScores.treeNum[i].foldScores[fold].avg)
            plot_y[fold][std_indx][i] = np.copy(treeScores.treeNum[i].foldScores[fold].std)
            #print(plot_y[fold][avg_indx][i])

    
    for i in range(num_trees):
        if i == 0:
            num_estimators = i+1
        else:
            num_estimators = (i)*10    
        plot_x[i] = np.copy(num_estimators)
        
    plt.figure(figsize=(20,10))
    plt.figure(1)
    # multiple line plot
    for fold_num in (range(num_folds)):
    #     print(plot_y[fold_num][avg_indx][:])
        plt.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:], label='Test Fold {0}'.format(fold_num))    
    plt.xlabel('Number of Trees',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    plt.title('Pavia Dataset Random Trees Classification',fontsize=20)
    plt.legend()
    plt.show()
    
    end = time.time()
    print("Random Forrest Runtime: {} seconds".format(end - start))


def knn_test(data, ground_truth, kmax):
    start = time.time()
    '''
        kmax: Maximum number to iterate up to in knn
    '''
    
    #Then, set up parameters for making figures
    h = .02  # step size in the mesh
    
    uniform_scores = np.zeros(kmax, dtype=np.float64)
    weighted_scores = np.zeros(kmax, dtype=np.float64)
    
    #Next set up the classifiers we will run
    #In this case, we are running K-NN with uniform distance (standard approach) and with weighted distances
    
    for i in tqdm(range(kmax)): #exclusive of the 5
        
        X = data
        Y = ground_truth
            
        n_neighbors = i+1
        
        classifiers = []
        classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
        classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
        names = ['K-NN_Uniform', 'K-NN_Weighted']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)
        
        
        # Iterate over classifiers
        for name, clf in zip(names, classifiers): 
    
            #Train the classifier
            clf.fit(X_train, y_train)
    
            #Test the classifier
            if name == 'K-NN_Uniform':
                uniform_scores[i] = (clf.score(X_test, y_test))
            else:
                weighted_scores[i] = (clf.score(X_test, y_test))
                        
    figure = plt.figure()
    plt.scatter(x=np.arange(start=1, stop=kmax+1), y=uniform_scores)
    plt.title("K from 1 to {}: Uniformed K-NN".format(kmax))
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    
    figure = plt.figure()
    plt.scatter(x=np.arange(start=1, stop=kmax+1), y=weighted_scores)
    plt.title("K from 1 to {}: Weighted K-NN".format(kmax))
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    plt.show();
    end = time.time()
    print("Knn Runtime: {} seconds".format(end - start))

if __name__ == "__main__":
    image_arr, image_class_arr, image_list, image_class = load_images()
#    randfom_forest_test(image_arr, image_class_arr, 20)
#    knn_test(image_arr, image_class_arr, 5)
    f = 1