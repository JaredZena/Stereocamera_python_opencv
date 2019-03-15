# TOMATO SEGMENTATION
import glob
import cv2
import numpy as np
img_path = 'C:/Users/Jared/Documents/python_scripts/'
#images_left = glob.glob(img_path + 'test1_rectified_images_left/*.PNG')
#images_right = glob.glob(img_path + 'test1_rectified_images_right/*.PNG')
images_left = glob.glob(img_path + 'left_test_1/*.PNG')
images_right = glob.glob(img_path + 'right_test_1/*.PNG')
images_left.sort()
images_right.sort()
erosion_kernel = np.ones((3, 3), np.uint8)
blur_kernel = (1, 1)
minup = (165, 40, 170)
maxup = (185, 255, 255)
minlow = (0, 55, 200)
maxlow = (20, 255, 255)
for i, fname in enumerate(images_right):
    img2 = cv2.imread(images_right[i])
    img1 = cv2.imread(images_left[i])
    cv2.imshow("Original Left Image", img1)
    cv2.imshow("Original RIght Image", img2)
    cv2.waitKey(800)
    # convert to hsv
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # mask of red (36,0,0) ~ (255, 255,70)
    mask_upper1 = cv2.inRange(hsv1, minup, maxup)
    mask_lower1 = cv2.inRange(hsv1, minlow, maxlow)
    mask_upper2 = cv2.inRange(hsv2, minup, maxup)
    mask_lower2 = cv2.inRange(hsv2, minlow, maxlow)
    imask_upper1 = mask_upper1 > 0
    imask_lower1 = mask_lower1 > 0
    imask_upper2 = mask_upper2 > 0
    imask_lower2 = mask_lower2 > 0

    # slice the red upper hue
    red_upper1 = np.zeros_like(img1, np.uint8)
    red_lower1 = np.zeros_like(img1, np.uint8)
    red_upper1[imask_upper1] = img1[imask_upper1]
    red_lower1[imask_lower1] = img1[imask_lower1]
    # slice the red upper hue
    red_upper2 = np.zeros_like(img2, np.uint8)
    red_lower2 = np.zeros_like(img2, np.uint8)
    red_upper2[imask_upper2] = img2[imask_upper2]
    red_lower2[imask_lower2] = img2[imask_lower2]
    """red_upper1 = cv2.inRange(hsv1, minup, maxup)
    red_lower1 = cv2.inRange(hsv1, minlow, maxlow)
    red_upper2 = cv2.inRange(hsv2, minup, maxup)
    red_lower2 = cv2.inRange(hsv2, minlow, maxlow)"""
    # Add both images
    red1 = cv2.addWeighted(red_upper1, 1, red_lower1, 1, 0, None)
    red2 = cv2.addWeighted(red_upper2, 1, red_lower2, 1, 0, None)

    # Eroding clusters
    red_eroded1 = cv2.erode(red1, erosion_kernel, iterations=1)
    red_eroded2 = cv2.erode(red2, erosion_kernel, iterations=1)

    # Applying gaussian filter
    red_blured1 = cv2.GaussianBlur(red_eroded1, blur_kernel, 0)
    red_blured2 = cv2.GaussianBlur(red_eroded2, blur_kernel, 0)

    #cv2.imwrite(img_path + 'test1_segmented_images_left/' + 'rectandsegm_left_' + '{}'.format(i) + '.png', red_eroded1)
    #cv2.imwrite(img_path + 'test1_segmented_images_right/' + 'rectandsegm_right_' + '{}'.format(i) + '.png', red_eroded2)
    cv2.imwrite(img_path + 'only_segmented_left_test1/' + 'segmented_left_' + '{}'.format(i) + '.png', red_eroded1)
    cv2.imwrite(img_path + 'only_segmented_right_test1/' + 'segmented_right_' + '{}'.format(i) + '.png', red_eroded2)
    cv2.imshow('Left Segmented', red_blured1)
    cv2.imshow('Right Segmented', red_blured2)
    #cv2.imshow('Left Lower', red_lower2)
    #cv2.imshow('Right Upper', red_upper2)

print('The segmented images have been saved succesfully!')
cv2.waitKey(0)
cv2.destroyAllWindows()
