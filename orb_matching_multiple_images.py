    # Stereo matcher
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    cv2.waitKey(100)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Find the keypoints and the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    # Find matches and sort them based on accuracy
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Show
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(img3)
    plt.savefig(img_path + 'orb_matched_images_test1/' + 'orb_matched_' + '{}'.format(i) + '.png')
    print(i)
cv2.waitKey(0)
cv2.destroyAllWindows()
