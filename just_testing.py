#JUST TESTING

import numpy as np
import cv2

# Define checkboard size
cols = 9
rows = 6
checkboard_size = (cols,rows)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp = np.zeros((cols*rows, 3), np.float32)
#objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
#objpoints = []  # 3d point in real world space

# Reading image
img = cv2.imread("C:\\Users\\Jared\\Documents\\python_scripts\\opencv_frame_0.png")
cv2.imshow("imagen",img)

# Grayscaling image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)

#Finding Chessboard corners
ret, corners = cv2.findChessboardCorners(gray, checkboard_size,
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH+
                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
print(ret)
print(corners)

# Draw and display the corners
if ret is True:

        ret = cv2.drawChessboardCorners(img, checkboard_size,
                                                  corners, ret)
        cv2.imshow("Image with corners", img)
        img_name = "Image_with_corners.png"
        cv2.imwrite(img_name, img)
        print(ret)

