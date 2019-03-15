# Stereo matcher
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_left/rectified_left_8.png', 0)
img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_right/rectified_right_8.png', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Find the keypoints and the descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Find matches and sort them based on accuracy
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print(des1)
print(des2)
# Show
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
plt.imshow(img3)
plt.show()
