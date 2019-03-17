# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:20:27 2019

@author: tdtra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from PIL import Image
import glob
imageList = []
for filename in glob.glob('C:/Users/tdtra/Downloads/2019_sp_ml_train_data/2019_sp_ml_train_data/a/*.jpeg'): #assuming gif
    img = Image.open(filename)
    imageList.append(img)

np.save('a.npy', imageList)


