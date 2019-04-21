from skimage import io
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

filename = '/Users/applemac/Documents/GitHub/project-trm/TRM_Pics/Nikita/A/IMG_2297 resized.jpg'
img1 = io.imread(filename)
img1= rgb2gray(img1)

tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                           translation=(0, -200))
img3 = tf.warp(img1, tform)

descriptor_extractor = ORB(n_keypoints=200)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

# descriptor_extractor.detect_and_extract(img3)
# keypoints3 = descriptor_extractor.keypoints
# descriptors3 = descriptor_extractor.descriptors

# matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

# plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")


plt.show()