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

def knn_classifier(data_arr, class_arr,  train_size, title, k_max=10):
    '''
    Fit and test a KNN classifier on input data
    data_arr should be [N x M] where N is data points and M is features
    class_arr should be [N x 1]
    k_max: Maximum number to iterate up to in knn
    '''

    start_time = time.time()

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
        X_train, X_test, y_train, y_test = train_test_split(data_arr, class_arr, test_size=train_size)

        # Iterate over classifiers
        for name, clf in zip(names, classifiers): 

            #Train the classifier
            clf.fit(X_train, y_train)

            #Test the classifier
            if name == 'K-NN_Weighted':
                weighted_scores[i] =  (clf.score(X_test, y_test))

    plt.figure()
    plt.scatter(x=np.arange(start=1, stop=k_max+1), y=weighted_scores)
    plt.title("Weighted Knn from 1 to {} {}".format(k_max,title))
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    
    if os.path.exists('Figures/result.png'):
        plt.savefig('Figures/result_{}.png'.format(int(time.time())))
    else:
        plt.savefig('Figures/result.png')

    end_time = time.time()
    print("Knn Runtime: {} seconds".format(end_time - start_time))

    f = 1

def random_forest_classifier(data_arr, class_arr, folds, title, max_trees=50):
    '''
    Fit and test a random forest classifier on input data
    data_arr should be [N x M] where N is data points and M is features
    class_arr should be [N x 1]
    Max Trees must be a multiple of 10
    '''
    num_trees = math.floor(max_trees/10)+1
    start_time = time.time()
    num_folds = folds
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
                                        n_estimators=num_estimators, n_jobs=-1, max_features=50)
        modelTrained = forest.fit(data_arr, class_arr)

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

    # Multiple line plot
    for fold_num in range(num_folds):
        # print(plot_y[fold_num][avg_indx][:])
        plt.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:],
                     label='Test Fold {0}'.format(fold_num))    

    plt.xlabel('Number of Trees', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Random Forests Classification {}'.format(title))
    plt.legend()
    
    if os.path.exists('Figures/result.png'):
        plt.savefig('Figures/result_{}.png'.format(int(time.time())))
    else:
        plt.savefig('Figures/result.png')
    
    end_time = time.time()
    print("Random Forest Runtime: {} seconds".format(end_time - start_time))
    return modelTrained
    
def random_forest_feat_importance(data_arr, class_arr):
    ##############################Feature Importance##############################
    forest = RandomForestClassifier(criterion='entropy', n_jobs=-1)
    forest.fit(data_arr, class_arr)
    importances = forest.feature_importances_
    
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
#    print("Feature ranking:")
    
    
    text_file = open("Output.txt", "w")
    for f in range(data_arr.shape[1]):
        temp = ("%d. feature %d (%f)\n" % (f + 1, indices[f], importances[indices[f]]))
        text_file.write(temp)
    text_file.close()
    
#    plt.figure(figsize=(75,75))
#    plt.bar(range(data_arr.shape[1]), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    plt.xticks(range(1, data_arr.shape[1]+1, 10) )
#    plt.xlabel("Features by Ranking")
#    plt.xlim([-1, data_arr.shape[1]])
#    plt.title("Feature importances")
#    plt.rc('font', size=150)          # controls default text sizes
#    plt.show()
    

    
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

def AFTest(model):
    paths = ["2019_sp_ml_train_data", "A_n_F", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
    rgb_image_arr_af, hsv_image_arr_af, image_class_arr_af = load_images(paths[1])
    
    hsv_image_arr_af = preprocess_all_images(hsv_image_arr_af)
    hsv_image_arr_feats_af = extract_features_all_images(hsv_image_arr_af)
    hsv_image_arr_af = hsv_image_arr_af.reshape(len(hsv_image_arr_af),10000)
    hsv_image_arr_af = np.concatenate((hsv_image_arr_af, hsv_image_arr_feats_af),axis=1)       
    
    hsv_image_arr_af, image_class_arr_af = shuffle_in_unison(hsv_image_arr_af, image_class_arr_af)
    print(paths[1], 'score: ', model.score(hsv_image_arr_af, image_class_arr_af))
    return (randomForestModel.predict(test_data)).tolist()

    
if __name__ == "__main__":
    # # Run this code only on ubuntu
    # if os.name == "posix":
    #     # Change the method with which new threads are created, required for sklearn multithreading
    #     # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    #     import multiprocessing
    #     multiprocessing.set_start_method('forkserver')

#    rgb_image_arr, hsv_image_arr, image_class_arr = load_images()
#    
#    print("RGB KNN No Pre-Processing")
#    rgb_image_arr = rgb_image_arr.reshape(len(rgb_image_arr),30000)    
#    knn_classifier(rgb_image_arr, image_class_arr)
#    
#    print("HSV KNN No Pre-Processing")
#    hsv_image_arr = hsv_image_arr.reshape(len(hsv_image_arr),30000)
#    knn_classifier(hsv_image_arr, image_class_arr)

    paths = ["2019_sp_ml_train_data", "A_n_F", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
    rgb_image_arr, hsv_image_arr, image_class_arr = load_images(paths[0])
    
#    print("RGB KNN")
#    rgb_image_arr = preprocess_all_images(rgb_image_arr)
#    rgb_image_arr_feats = extract_features_all_images(rgb_image_arr)
#    rgb_image_arr = rgb_image_arr.reshape(len(rgb_image_arr),10000)
#    rgb_image_arr = np.concatenate((rgb_image_arr, rgb_image_arr_feats),axis=1)  
#    knn_classifier(rgb_image_arr, image_class_arr)
    
    hsv_image_arr = preprocess_all_images(hsv_image_arr)
    hsv_image_arr_feats = extract_features_all_images(hsv_image_arr)
    hsv_image_arr = hsv_image_arr.reshape(len(hsv_image_arr),10000)
    hsv_image_arr = np.concatenate((hsv_image_arr, hsv_image_arr_feats),axis=1)
    
##    print("Running Different Combos of Data")
##    for i in range(1,len(paths)):
##        for j in range(3):
##            print(("Run: {} {}").format(paths[i],j))
##            knn_classifier(hsv_image_arr, image_class_arr, 0.7, ("Run: {} {}").format(paths[i],j))
##            random_forest_classifier(hsv_image_arr, image_class_arr, 10, ("Run: {} {}").format(paths[i],j))
#    
    cross_val_folds = 10
    print("Shuffling data and use 70% for training")
    for i in range(1):
        print(("Run: {}").format(i))
        hsv_data, hsv_gt = shuffle_in_unison(hsv_image_arr, image_class_arr)
#        knn_classifier(hsv_data, hsv_gt, 0.9, "Shuffling Data Run {}".format(i))
        randomForestModel = random_forest_classifier(hsv_data, hsv_gt, cross_val_folds, "Shuffling Data Run {}".format(i))
        
    plt.show()
    
    af_prediction_labels = AFTest(randomForestModel)
    with open("trmAFLabels.txt", "w") as file:
        file.write(str(af_prediction_labels))

''' Helpful commands:
        Display Image:
            plt.imshow(np.reshape(image_arr[i][j],(100,100,3)))
'''
