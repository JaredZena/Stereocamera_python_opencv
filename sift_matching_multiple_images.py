# sift_matching
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

img_path = 'C:/Users/Jared/Documents/python_scripts/'
images_left = glob.glob(img_path + 'test1_segmented_images_left/*.PNG')
images_right = glob.glob(img_path + 'test1_segmented_images_right/*.PNG')
images_left.sort()
images_right.sort()

for i, fname in enumerate(images_right):
    img2 = cv2.imread(images_right[i])
    img1 = cv2.imread(images_left[i])
    #cv2.imshow("Original Left Image", img1)
    #cv2.imshow("Original RIght Image", img2)
    # cv2.waitKey(10)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    plt.imshow(img3)
    plt.savefig(img_path + 'sift_matched_images_test1/' + 'sift_matched_' + '{}'.format(i) + '.png')
    print(i)
cv2.waitKey(100)
cv2.destroyAllWindows()
