import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('/Users/applemac/Documents/GitHub/project-trm/TRM_Pics/Nikita/A/IMG_2297 resized.jpg',0)

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
# print(kp_array)


# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)

# img2 = img.copy()
# for marker in kp:
# 	img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

# plt.imshow(img2),plt.show()