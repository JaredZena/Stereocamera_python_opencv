import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_left/rectified_left_20.png')
img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_right/rectified_right_20.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_left', img1)
cv2.imshow('gray_right', img2)
corners1 = cv2.goodFeaturesToTrack(gray1, 15, 0.07, 40)
corners2 = cv2.goodFeaturesToTrack(gray2, 15, 0.07, 40)
corners1 = np.int0(corners1)
print('corners1 = ', corners1)
corners2 = np.int0(corners2)
print('corners2 = ', corners2)

for i in corners1:
    # print('i= ', i)
    x, y = i.ravel()
    cv2.circle(img1, (x, y), 3, 255, -1)

for j in corners2:
    x2, y2 = j.ravel()
    cv2.circle(img2, (x2, y2), 3, 255, -1)

# plt.imshow(img1)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
ax1.imshow(img2)
ax2.imshow(img1)
plt.show()
