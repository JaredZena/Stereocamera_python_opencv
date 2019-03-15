# getting_sift_coord.py
#!/usr/bin/python3
# 2017.10.06 22:36:44 CST
# 2017.10.06 23:18:25 CST

"""
Environment:
    OpenCV 3.3  + Python 3.5

Aims:
(1) Detect sift keypoints and compute descriptors.
(2) Use flannmatcher to match descriptors.
(3) Do ratio test and output the matched pairs coordinates, draw some pairs in purple .
(4) Draw matched pairs in blue color, singlepoints in red.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_left/rectified_left_12.png')
img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_right/rectified_right_12.png')
# Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
#orb = cv2.ORB_create()

# Create flann matcher
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})

## Detect and compute
#img1 = cv2.imread(imgname)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kpts1, descs1 = sift.detectAndCompute(gray1, None)

# As up
#img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

# Ratio test
matches = matcher.knnMatch(descs1, descs2, 2)
matchesMask = [[0, 0] for i in range(len(matches))]
pn1 = (0, 0)
pn2 = (0, 0)
x1 = []
y1 = []
x2 = []
y2 = []
d = []
xy_array = []
sc = 0
for i, (m1, m2) in enumerate(matches):
    if m1.distance < 0.9 * m2.distance:
        # Notice: How to get the index
        pt1 = kpts1[m1.queryIdx].pt
        pt2 = kpts2[m1.trainIdx].pt
        if pt1 != pn1 and pt2 != pn2:
            if abs(pt2[1] - pt1[1]) < 5:
                pn1 = pt1
                pn2 = pt2
                print(sc, pn1, pn2)
                sc += 1
                x1.append(pn1[0])
                y1.append(pn1[1])
                x2.append(pn2[0])
                y2.append(pn2[1])
                d.append(abs(pn2[0] - pn1[0]))
                xy_array.append([pn1[0], pn1[1]])
                xy_array.append([pn2[0], pn2[1]])
                matchesMask[i] = [1, 0]

# Draw match in blue, error in red
print(d)
print(len(d))
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=0)

res = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, matches, None, **draw_params)
cv2.imshow("Result", res)

xy_array = np.asarray(xy_array)
print(xy_array)
plt.scatter(xy_array[:, 0], xy_array[:, 1])
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
