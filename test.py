######################################### Load Labels Imports ###################################
# Basic libs
import os
import glob
import time
from timeit import default_timer as timer
import math
import seaborn as sns
import string
from tqdm import tqdm

# Data science tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Pytorch
from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torch.utils.data import DataLoader, sampler
from torch.autograd import Variable

# Image manipulations
from PIL import Image, ImageFilter, ExifTags

######################################## Fast AI Imports ########################################
from fastai.vision import *

#################################################################################################
def load_labels(path):
    '''
    Load all images as RGB and HSV data in tensors
    Pair all images with classsification
    '''

    # Image manipulations
    from PIL import Image, ImageFilter, ExifTags

    # Record time to load images
    start_time = timer()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letter_dirs = glob.glob('{}/data/{}/*'.format(script_dir, path))
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

    end_time = timer()
    print("Labels loaded in ", end_time - start_time, "s")

    return (rgb_image_arr, hsv_image_arr, image_class_arr)


def load_images(easy=1):
    path = Path('data/')
    classes = ['a','b','c', 'd', 'e', 'f', 'g', 'h', 'i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    np.random.seed(42)
    if (easy):
        data = ImageDataBunch.from_folder(path, train='train', valid='valid', test='testAF',ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    else:
        data = ImageDataBunch.from_folder(path, train='train', valid='valid', test='test',ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    # data.show_batch(rows=3, figsize=(7,8))
#    print(data.classes, data.c, len(data.train_ds), len(data.valid_ds), len(data.test_ds))
    return data

def train_cnn(data):
    # can use resnet50 for more layers
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(3)
    learn.save('stage-1')

    learn.unfreeze()
    learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
    learn.save('stage-2')
    return learn

def load_cnn():
    path = Path('data/')
    learn = load_learner(path)
#    img = open_image(path/'test'/'V'/'IMG_2501 resized.jpg')
#    pred_class,pred_idx,outputs = learn.predict(img)
    # Print the predition
#    print(pred_class)
    return learn

def do_prediction(learner, data):
    preds = []
    for i in range(len(data.test_ds.x)):
        p = learner.predict(data.test_ds.x[i])
        preds.append(str(p[0]).lower())
    return preds
    
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def genLabels():
    #For Load
    easyRGB, easyHSV, easyLabels = load_labels("testAF")
    with open("easyLabels.txt", "w") as file:
        file.write(str(easyLabels.tolist()))
    
    hardRGB, hardHSV, hardLabels = load_labels("test")
    with open("hardLabels.txt", "w") as file:
        file.write(str(hardLabels.tolist()))
        
def genPredictions(easy=1):
   data = load_images(easy);
   estimatedLabels = do_prediction(learner, data)
   
   with open("estimatedLabels.txt", "w") as file:
       file.write(str(estimatedLabels))
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        defaults.device = torch.device("cuda")
    else:
        defaults.device = torch.device("cpu")
        
    learner = load_cnn()
    
    genLabels() # generate label files
    
    easy = 1
    genPredictions(easy)    # generate easy prediction files
    
    easy = 0
    genPredictions(easy)    # generate hard prediction files
    
       
    
            
    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # interp.plot_top_losses(9, figsize=(10,10))

# Testing
# This will create a file named 'export.pkl' in the directory 
# where we were working that contains everything we need to deploy 
# our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).

