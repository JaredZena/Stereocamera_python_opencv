# disparity_map_single_images.py
import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/only_segmented_left_test1/segmented_left_16.png')
img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/only_segmented_right_test1/segmented_right_16.png')
print(img1.shape)
print(img2.shape)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
0
stereo = cv2.StereoBM_create(numDisparities=80, blockSize=5)
#stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=16, SADWindowSize=15)
disparity = stereo.compute(gray1, gray2)
plt.imshow(disparity, 'gray')
plt.show()
