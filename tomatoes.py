# TOMATO SEGMENTATION
import glob
import cv2
import numpy as np
img_path = 'C:/Users/Jared/Documents/python_scripts/'
images_right = glob.glob(img_path + 'right_test_1/*.PNG')
images_left = glob.glob(img_path + 'left_test_1/*.PNG')
images_left.sort()
images_right.sort()
erosion_kernel = np.ones((3, 3), np.uint8)
for i, fname in enumerate(images_right):
    img2 = cv2.imread(images_right[i])
    img1 = cv2.imread(images_left[i])
    cv2.imshow("Original Left Image", img1)
    cv2.imshow("Original RIght Image", img2)
    cv2.waitKey(100)

    # convert to hsv
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # mask of red (36,0,0) ~ (255, 255,70)
    mask_upper1 = cv2.inRange(hsv1, (160, 100, 100), (179, 255, 255))
    mask_lower1 = cv2.inRange(hsv1, (0, 100, 100), (22, 255, 255))
    mask_upper2 = cv2.inRange(hsv2, (160, 100, 100), (179, 255, 255))
    mask_lower2 = cv2.inRange(hsv2, (0, 100, 100), (22, 255, 255))
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
    # Add both images
    red1 = red_upper1
    red2 = red_upper1
    cv2.addWeighted(red_upper1, 1, red_lower1, 1, 0, red1)
    cv2.addWeighted(red_upper2, 1, red_lower2, 1, 0, red2)

    # Eroding clusters
    red_eroded1 = cv2.erode(red1, erosion_kernel, iterations=1)
    red_eroded2 = cv2.erode(red2, erosion_kernel, iterations=1)

    # Applying gaussian filter
    red_blured1 = cv2.GaussianBlur(red_eroded1, (5, 5), 0)
    red_blured2 = cv2.GaussianBlur(red_eroded2, (5, 5), 0)

    # Display
    cv2.imshow("red_upper", red_upper1)
    cv2.imshow("red_lower", red_lower1)
    cv2.imshow("red_final", red1)
    cv2.imshow("red_blur", red_blured1)
    cv2.imshow("red_erosion", red_erosion1)

    cv2.imwrite(img_path + 'test1_rectified_images_left/' + 'rectified_left_' + '{}'.format(i) + '.png', dst1)
    cv2.imwrite(img_path + 'test1_rectified_images_right/' + 'rectified_right_' + '{}'.format(i) + '.png', dst2)
    cv2.imshow('Left Rectified', dst1)
    cv2.imshow('Right Rectified', dst2)
cv2.waitKey(0)
print('The rectified and undistorted images have been saved succesfully!')
cv2.destroyAllWindows()
