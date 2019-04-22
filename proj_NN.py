'''
NN Playground for classification of ASL using transfer learning from various pretrained models
'''

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
# # Useful for examining network
# from torchsummary import summary

def load_images(path):
    '''
    Load all images as RGB and HSV data in tensors
    Pair all images with classsification
    '''

    # Record time to load images
    start_time = timer()

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

            # # Rotate images properly based on EXIF data
            # try:
            #     for orientation in ExifTags.TAGS.keys():
            #         if ExifTags.TAGS[orientation] == 'Orientation':
            #             break
            #     exif = dict(im._getexif().items())

            #     if exif[orientation] == 3:
            #         im = im.rotate(180, expand=True)
            #     elif exif[orientation] == 6:
            #         im = im.rotate(270, expand=True)
            #     elif exif[orientation] == 8:
            #         im = im.rotate(90, expand=True)

            # except (AttributeError, KeyError, IndexError):
            #     # Cases: image doesn't have EXIF data
            #     pass

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
    print("Images loaded in ", end_time - start_time, "s")

    return (rgb_image_arr, hsv_image_arr, image_class_arr)

class Net(nn.Module):
    '''
    Neural net definition
    '''

    def __init__(self):
        super(Net, self).__init__()

        # Input = [batch_size, 3, 100, 100]
        self.netflow1 = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7), # [batch_size, 64, 94, 94]
                            nn.ReLU(inplace=True), # unchanged
                            nn.MaxPool2d(kernel_size=2), # [batch_size, 64, 47, 47]
                            
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7), # [batch_size, 64, 41, 41]
                            nn.ReLU(inplace=True), # unchanged
                            nn.MaxPool2d(kernel_size=2), # [batch_size, 64, 20, 20]
                            
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7), # [batch_size, 64, 14, 14]
                            nn.ReLU(inplace=True), # unchanged
                            nn.MaxPool2d(kernel_size=2), # [batch_size, 64, 7, 7]
                            
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7), # [batch_size, 64, 2, 2]
                            nn.ReLU(inplace=True), # unchanged
                            nn.MaxPool2d(kernel_size=2)) # [batch_size, 64, 1, 1]

        # Uses flattened data as input
        self.netflow2 = nn.Sequential(
                            nn.Linear(64, 128), # [batch_size, 128]
                            nn.ReLU(inplace=True), # unchanged
                            nn.Dropout(0.4), # unchanged
                            nn.Linear(128, 24), # [batch_size, 24]
                            nn.Softmax(1)) # unchanged

    def forward(self, x):
        '''
        Custom forward propagation algorithm
        '''

        x = self.netflow1(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.netflow2(x)

        return x

    def test(self, predictions, labels):
        '''
        Test the NN
        '''

        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1

        acc = correct / len(predictions)
        print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))


    def evaluate(self, predictions, labels):
        '''
        Evaluate the NN
        '''

        #print('\nPrediction Labels: ', predictions)
        #print('\nFunction Labels: ', labels)

        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        return(acc)

def train_net(training_batch, training_batch_labels, testing_batch, testing_batch_labels):
    '''
    Train NN
    '''

    # Record time to train net
    start_time = timer()

    # Set optimizer and loss function
    optimizer = optim.Adam(net.parameters(), learning_rate)
    loss_func = nn.CrossEntropyLoss()

    loss_log = []
    acc_log = []

    for e in tqdm(range(epochs)):
        # idx = torch.randperm(x.nelement())
        # x = x.view(-1)[idx].view(x.size())
        # y = y.view(-1)[idx].view(y.size())

        # Train the net with mini-batches
        for i in range(0, training_batch.shape[0], mini_batch_size):
            x_mini = training_batch[i:i + mini_batch_size] 
            y_mini = training_batch_labels[i:i + mini_batch_size] 

            optimizer.zero_grad()
            net_out = net(Variable(x_mini))
            loss = loss_func(net_out, Variable(y_mini))
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                #pred = net(Variable(test_data_formated))
                loss_log.append(loss.item())
                acc_log.append(net.evaluate(torch.max(net(Variable(testing_batch[:500])).data, 1)[1],
                                            testing_batch_labels[:500]))

        print('Epoch: {} - Loss: {:.6f}, Accuracy: {:.1f}%'.format(e + 1, loss.item(), 100*acc_log[-1]))

    plt.figure(figsize=(10, 8))
    plt.plot(loss_log[2:])
    plt.plot(acc_log)
    plt.plot(np.ones(len(acc_log)), linestyle='dashed')
    plt.show()

    end_time = timer()
    print("CNN trained in ", end_time - start_time, "s")


    ff = 1

def test_net(testing_batch):
    '''
    Test NN
    '''

    testing_batch_labels = []

    return testing_batch_labels


if __name__ == "__main__":
    # Check for CUDA availability to later push net and tensors to it
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create the NN
    net = Net()
    net.to(device)
    print(net)

    # Define hyperparameters
    epochs = 250
    mini_batch_size = 200
    learning_rate = 1e-4

    # Load images
    paths = ["2019_sp_ml_train_data", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
    rgb_image_arr, hsv_image_arr, image_class_arr = load_images(paths[0])

    # # Test print to ensure proper loading
    # trouble_img_arr = rgb_image_arr[1,:,:,:]
    # plt.imshow(trouble_img_arr)
    # plt.show()

    rgb_image_arr = np.transpose(rgb_image_arr, (0, 3, 1, 2))
    hsv_image_arr = np.transpose(hsv_image_arr, (0, 3, 1, 2))

    # Define alphabet lookup table
    alph = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i',
            9:'k', 10:'l', 11:'m', 12:'n', 13:'o', 14:'p', 15:'q', 16:'r',
            17:'s', 18:'t', 19:'u', 20:'v', 21:'w', 22:'x', 23:'y'}
    alph = dict((v, k) for k, v in alph.items())

    image_class_nums_arr = []
    for i in range(len(image_class_arr)):
        #print(alph[image_class_arr[i]])
        image_class_nums_arr.append(alph[image_class_arr[i]])
    image_class_nums_arr = np.asarray(image_class_nums_arr, dtype=np.uint8)

    # rgb_image_train, rgb_image_test, train_labels, test_labels = train_test_split(rgb_image_arr, image_class_nums_arr, test_size=0.33, random_state=42)
    train_arr, test_arr, train_labels, test_labels = train_test_split(rgb_image_arr, image_class_nums_arr, test_size=0.33)

    # Convert data from numpy arrays to tensors (datatypes changed due to model requirements)
    train_tensor = torch.FloatTensor(train_arr).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    test_tensor = torch.FloatTensor(test_arr).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)

    # Normalize data to (-1, 1]
    train_tensor = (train_tensor - 127) / 128
    train_labels_tensor = (train_tensor - 127) / 128
    test_tensor = (train_tensor - 127) / 128
    test_labels_tensor = (train_tensor - 127) / 128

    # Print 
    # print('\nTensor Data: ', train_tensor)
    # print('\nTensor Data Shape: ', train_tensor.shape)
    # print('\nTensor Labels: ', train_labels_tensor)
    # print('\nTensor Labels Shape: ', train_labels_tensor.shape)

    # # Test print an image from tensors
    # trouble_img_tensor = train_tensor[1, :, :, :].cpu()
    # trouble_img_tensor = trouble_img_tensor.numpy() / 255
    # trouble_img_tensor = np.transpose(trouble_img_tensor, (1, 2, 0))
    # plt.imshow(trouble_img_tensor)
    # plt.show()

    train_net(train_tensor, train_labels_tensor, test_tensor, test_labels_tensor)
