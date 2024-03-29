# flann_matching.py
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
    cv2.imshow("Original Left Image", img1)
    cv2.imshow("Original RIght Image", img2)
    cv2.waitKey(10)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for j, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[j] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3)
    plt.savefig(img_path + 'flann_matched_images_test1/' + 'flann_matched_' + '{}'.format(i) + '.png')
    print(i)
cv2.waitKey(100)
cv2.destroyAllWindows()
