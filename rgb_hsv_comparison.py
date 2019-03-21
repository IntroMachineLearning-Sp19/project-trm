import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2


im = Image.open("/home/michael/Documents/EEL4930-ML/Project/2019_sp_ml_train_data/b/1IMG_3199.jpeg")
img_rgb = np.asarray(list(im.getdata())) # a set of 3 values(R, G, B)

img_hsv = np.asarray(list(im.convert('HSV').getdata()))

img_rbg_2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


plt.figure()
plt.imshow(img_rgb)
plt.show()

plt.figure()
plt.imshow(img_hsv)
plt.show()

plt.figure()
plt.imshow(img_rbg_2)
plt.show()
