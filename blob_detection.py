# blob_detection.py
# Standard imports
import cv2
import numpy as np
import glob

img_path = 'C:/Users/Jared/Documents/python_scripts/'
images_left = glob.glob(img_path + 'test1_segmented_images_left/*.PNG')
images_right = glob.glob(img_path + 'test1_segmented_images_right/*.PNG')
images_left.sort()
images_right.sort()
print(len(images_left))
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 20
params.maxThreshold = 1000
# Filter by Area.
params.filterByArea = True
params.minArea = 30
params.maxArea = 5000
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.3
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

blank_image = np.zeros((480, 640), np.uint8)
for i, fname in enumerate(images_right):
    img2 = cv2.imread(images_right[i])
    img1 = cv2.imread(images_left[i])
    cv2.imshow("Original Segmented Left", img1)
    cv2.imshow("Original Segmented RIght", img2)
    cv2.waitKey(800)
# Read image
#img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_left/rectified_left_15.png', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_right/rectified_right_15.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.bitwise_not(img1)
    img2 = cv2.bitwise_not(img2)

    img1_pt = cv2.bitwise_not(blank_image)
    img2_pt = cv2.bitwise_not(blank_image)

    # Detect blobs.
    keypoints1 = detector.detect(img1)
    x_list1 = []
    y_list1 = []
    d_list1 = []
    for i in range(len(keypoints1)):
        x1 = keypoints1[i].pt[0]
        y1 = keypoints1[i].pt[1]
        d1 = keypoints1[i].size
        x_list1.append(x1)
        y_list1.append(y1)
        d_list1.append(d1)
        print(str(x1) + " , " + str(y1) + " , " + str(d1))
        cv2.circle(img1_pt, (int(x1), int(y1)), 2, (0, 0, 0), -1)

    keypoints2 = detector.detect(img2)
    x_list2 = []
    y_list2 = []
    d_list2 = []
    for j in range(len(keypoints2)):
        x2 = keypoints2[j].pt[0]
        y2 = keypoints2[j].pt[1]
        d2 = keypoints2[j].size
        x_list2.append(x2)
        y_list2.append(y2)
        d_list2.append(d2)
        print(str(x2) + " , " + str(y2) + " , " + str(d2))
        cv2.circle(img2_pt, (int(x2), int(y2)), 2, (0, 0, 0), -1)

    cv2.imshow('tomato_points_right_' + '{}'.format(i), img1_pt)
    cv2.imshow('tomato_points_right_' + '{}'.format(j), img2_pt)
    #cv2.imwrite('C:/Users/Jared/Documents/python_scripts/tomato_points_left/puntos_left.png', img1_pt)
    #cv2.imwrite('C:/Users/Jared/Documents/python_scripts/tomato_points_right/puntos_right.png', img2_pt)
    cv2.imwrite(img_path + 'tomato_points_left/' + 'puntos_left_' + '{}'.format(i) + '.png', img1_pt)
    cv2.imwrite(img_path + 'tomato_points_right/' + 'puntos_right_' + '{}'.format(j) + '.png', img2_pt)
print('The images have been saved succesfully!')
cv2.waitKey(0)
cv2.destroyAllWindows()
