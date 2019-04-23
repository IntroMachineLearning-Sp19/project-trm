import os
import glob
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import seaborn as sns
import string
from tqdm import tqdm

from sklearn.model_selection import train_test_split
########################################################################################################    
if (torch.cuda.is_available()):
    cuda_available = 1
else:
    cuda_available = 0
########################################################################################################    
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
########################################################################################################    
alph = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:"f", 6:'g', 7:'h', 8:'i', 9:'k', 10:'l', 11:'m', 12:'n',
        13:'o', 14:'p', 15:'q', 16:'r', 17:'s', 18:'t', 19:'u', 20:'v', 21:'w', 22:'x', 23:'y'}
alph = dict((v,k) for k,v in alph.items())
########################################################################################################    
def reshape_to_2d(data, dim):
    reshaped = []
    for i in data:
        reshaped.append(i.reshape(1, dim, dim))

    return np.array(reshaped)

paths = ["2019_sp_ml_train_data", "Combined", "Combined_no_Michael", "Combined_no_Nikita", "Combined_no_Rosemond", "Combined_no_Trung"]    
rgb_image_arr, hsv_image_arr, image_class_arr = load_images(paths[1])
rgb_image_arr = rgb_image_arr.reshape(len(rgb_image_arr),3,100,100)

image_class_arr_nums = []
for i in range(len(image_class_arr)):
    #print(alph[image_class_arr[i]])
    image_class_arr_nums.append(alph[image_class_arr[i]])
rgb_image_arr, test_data, image_class_arr_nums, test_labels = train_test_split(rgb_image_arr, image_class_arr_nums, test_size=0.33, random_state=42)

if (cuda_available):
    x = torch.FloatTensor(rgb_image_arr).cuda()
else:
    x = torch.FloatTensor(rgb_image_arr)
    
#print('\nLabels: ', image_class_arr_nums)
if (cuda_available):
    y = torch.LongTensor(np.array(image_class_arr_nums, dtype=np.int64)).cuda()
else:
    y = torch.LongTensor(np.array(image_class_arr_nums, dtype=np.int64))
#print('\nTensor Labels: ', y)

if (cuda_available):
    test_data_formated = torch.FloatTensor(test_data).cuda()
else:
    test_data_formated = torch.FloatTensor(test_data)

epochs = 100
batch_size = 100
learning_rate = 0.005
########################################################################################################    
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print('\n', x.size())
        return x

class Network(nn.Module): 
    
    def __init__(self):
        super(Network, self).__init__()
        # Input = [batch_size, 1, 28, 28]
        self.netflow1 = nn.Sequential(
                #PrintLayer(),
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),  # [batch_size, 64, 26, 26]
                #PrintLayer(),
                nn.ReLU(inplace=True), # unchanged
                #PrintLayer(),
                nn.MaxPool2d(kernel_size=2), # [batch_size, 64, 13, 13]
                
                #PrintLayer(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), # [batch_size, 64, 11, 11]
                #PrintLayer(),
                nn.ReLU(inplace=True), # unchanged
                #PrintLayer(),
                nn.MaxPool2d(kernel_size=2), # [batch_size, 64, 5, 5]

                #PrintLayer(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), # [batch_size, 64, 3, 3]
                #PrintLayer(),
                nn.ReLU(inplace=True), # unchanged
                #PrintLayer(),
                nn.MaxPool2d(kernel_size=2)) # [batch_size, 64, 1, 1]

        # Uses flattened data as input
        self.netflow2 = nn.Sequential(
                #PrintLayer(),
                nn.Linear(6400, 128), # [batch_size, 128]
                #PrintLayer(),
                nn.ReLU(inplace=True), # unchanged
                #PrintLayer(),
                nn.Dropout(0.2), # unchanged
                #PrintLayer(),
                nn.Linear(128, 24), # [batch_size, 24]
                #PrintLayer(),
                nn.Softmax(dim=1)) # unchanged

# Rosemond
#    def __init__(self):
#        super(Network, self).__init__()
#        
#        self.netflow1 = nn.Sequential(
#                nn.Conv2d(3, 10, 3),
#                nn.ReLU(inplace=True),
#                nn.MaxPool2d(2),
#                nn.Conv2d(10, 20, 3),
#                nn.ReLU(inplace=True),
#                nn.MaxPool2d(2),
#                nn.Conv2d(20, 30, 3),
#                nn.ReLU(inplace=True),
#                nn.Dropout2d(),
#            )
#        
#        self.netflow2 = nn.Sequential(
#                nn.Linear(13230, 270),
#                nn.ReLU(inplace=True),
#                nn.Linear(270, 24),
#                nn.ReLU(inplace=True),
#                nn.LogSoftmax(dim=1)
#            )
    
    def forward(self, x):
        x = self.netflow1(x)
        x = x.view(x.size(0), -1)
        x = self.netflow2(x)
        return x
    
    def test(self, predictions, labels):
        
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))
        
    
    def evaluate(self, predictions, labels):
        #print('\nPrediction Labels: ', predictions)
        #print('\nFunction Labels: ', labels)
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        return(acc)
        
########################################################################################################            
net = Network()
if (cuda_available):
    net.cuda()
print(net)

optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.7)
loss_func = nn.CrossEntropyLoss()

loss_log = []
acc_log = []

for e in tqdm(range(epochs)):
    idx = torch.randperm(x.nelement())
    x = x.view(-1)[idx].view(x.size())
    
    idx = torch.randperm(y.nelement())
    y = y.view(-1)[idx].view(y.size())
    for i in range(0, x.shape[0], batch_size):
        x_mini = x[i:i + batch_size] 
        y_mini = y[i:i + batch_size] 
    
        optimizer.zero_grad()
        net_out = net(Variable(x_mini))
        loss = loss_func(net_out, Variable(y_mini))
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            #pred = net(Variable(test_data_formated))
            loss_log.append(loss.item())
            acc_log.append(net.evaluate(torch.max(net(Variable(test_data_formated[:500])).data, 1)[1], 
                                        test_labels[:500]))
        
    print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))
    
plt.figure(figsize=(10,8))
plt.plot(loss_log[2:])
plt.plot(acc_log)
plt.plot(np.ones(len(acc_log)), linestyle='dashed')

predictions = net(Variable(test_data_formated))
net.test(torch.max(predictions.data, 1)[1], test_labels)
########################################################################################################    