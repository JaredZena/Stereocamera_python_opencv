import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import cv2
n_clusters = 3


img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_right/rectified_right_13.png')
img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/test1_segmented_images_left/rectified_left_13.png')

#img1 = cv2.imread('C:/Users/Jared/Documents/python_scripts/only_segmented_right_test1/segmented_right_15.png')
#img2 = cv2.imread('C:/Users/Jared/Documents/python_scripts/only_segmented_left_test1/segmented_left_15.png')

cv2.imshow('Imagen 1', img1)
cv2.imshow('Imagen 2', img2)


# Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
# orb = cv2.ORB_create()

# Create flann matcher
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})

# Detect and compute
# img1 = cv2.imread(imgname)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kpts1, descs1 = sift.detectAndCompute(gray1, None)

# As up
# img2 = cv2.imread(imgname2)
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
xy_array1 = []
xy_array2 = []
sc = 0
for i, (m1, m2) in enumerate(matches):
    if m1.distance < .9 * m2.distance:
        # Notice: How to get the index
        pt1 = kpts1[m1.queryIdx].pt
        pt2 = kpts2[m1.trainIdx].pt
        if pt1 != pn1 and pt2 != pn2:
            if abs(pt2[1] - pt1[1]) < 6:
                if (gray1.item(int(pt1[1]), int(pt1[0])) >= 10 and gray2.item(int(pt2[1]), int(pt2[0])) >= 10):
                    pn1 = pt1
                    pn2 = pt2
                    print(sc, pn1, pn2)
                    sc += 1
                    x1.append(pn1[0])
                    y1.append(pn1[1])
                    x2.append(pn2[0])
                    y2.append(pn2[1])
                    d.append(abs(pn2[0] - pn1[0]))
                    xy_array1.append([pn1[0], pn1[1]])
                    xy_array2.append([pn2[0], pn2[1]])
                    matchesMask[i] = [1, 0]

# Draw match in blue, error in red
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=2)

res = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, matches, None, **draw_params)


x1 = np.asarray(xy_array1)
x2 = np.asarray(xy_array2)


clf1 = KMeans(n_clusters)
clf1.fit(x1)

clf2 = KMeans(n_clusters)
clf2.fit(x2)

centroids1 = clf1.cluster_centers_
labels1 = clf1.labels_
centroids2 = clf2.cluster_centers_
labels2 = clf2.labels_


# TOMATO CLUSTER MATCHING and DISPARITY CALCULATION
indexes = []
for m in range(len(centroids1)):
    dist_array = []
    for n in range(len(centroids2)):
        dist = math.sqrt((centroids2[n, 0] - centroids1[m, 0])**2 + (centroids2[n, 1] - centroids1[m, 1])**2)
        print('dist', dist)
        dist_array.append(dist)
    indexes.append(dist_array.index(min(dist_array)))

sorted_centroids1 = []
sorted_centroids2 = []
for index1 in indexes:
    for j3, (x22, y22) in enumerate(centroids2):
        if (j3 == index1):
            sorted_centroids2.append([x22, y22])
            print('j3', j3)

for i3, (x11, y11) in enumerate(centroids1):
    sorted_centroids1.append([x11, y11])

sorted_centroids1 = np.asarray(sorted_centroids1)
sorted_centroids2 = np.asarray(sorted_centroids2)

final_dist_array = []
for n2 in range(len(centroids2)):
    dist = math.sqrt((sorted_centroids2[n2, 0] - sorted_centroids1[n2, 0])**2 + (sorted_centroids2[n2, 1] - sorted_centroids1[n2, 1])**2)
    final_dist_array.append(dist)

cluster_disparity_array = []
for n3 in range(len(centroids2)):
    disparity = 631 + sorted_centroids2[n3, 0] - sorted_centroids1[n3, 0]
    cluster_disparity_array.append(disparity)

#xx1 = 470
#xx2 = 590
#yy1 = 0.45
#yy2 = 2.25

cluster_depth_array = []
for n4 in range(len(centroids2)):
    # depth = (((yy2 - yy1) / (xx2 - xx1)) * (cluster_disparity_array[n4] - xx1)) + yy1
    if (cluster_disparity_array[n4] > 580):
        cluster_depth_array.append(2.25)
    elif (cluster_disparity_array[n4] < 530):
        cluster_depth_array.append(0.45)
    else:
        cluster_depth_array.append(1.35)

# DISTANCE CALCULATION Z AXIS

# PLOTTING RESULTS
colors = ['g.', 'r.', 'c.', 'b.', 'k.', 'o.', 'y.', 'w.']
for i2 in range(len(x1)):
    plt.ylim(480, 0)
    plt.xlim(0, 640)
    plt.subplot(121)
    plt.plot(x1[i2][0], x1[i2][1], colors[labels1[1]], markersize=15)
    plt.scatter(centroids1[:, 0], centroids1[:, 1], marker='x', s=120, linewidths=5)

for j2 in range(len(x2)):
    plt.ylim(480, 0)
    plt.xlim(0, 640)
    plt.subplot(122)
    plt.plot(x2[j2][0], x2[j2][1], colors[labels2[1]], markersize=15)
plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='x', s=120, linewidths=5)

# vis = np.concatenate((img1, img2), axis=1)
# cv2.imshow(vis)
# PRINTING RESULTS
print(" ")
print('centroids1', centroids1)
print('centroids2', centroids2)

print('Sorted centroids1', sorted_centroids1)
print('Sorted centroids2', sorted_centroids2)
print(final_dist_array)
print(cluster_disparity_array)
print('cluster_depth_array', cluster_depth_array)
print(img1.shape)
reversed_array = [n5 for n5 in reversed(range(len(cluster_depth_array)))]
print('reversed_array = ', reversed_array)
font = cv2.FONT_HERSHEY_SIMPLEX
for n6 in range(len(cluster_depth_array)):
    cv2.putText(res, str(cluster_depth_array[n6]), (int(sorted_centroids1[n6, 0]), int(sorted_centroids1[n6, 1]) + 35), font, .8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(res, str(cluster_depth_array[n6]), (int(sorted_centroids2[n6, 0]) + img1.shape[1], int(sorted_centroids2[n6, 1]) + 35), font, .8, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow("Result", res)


# plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
