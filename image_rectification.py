# UNDISTORTING AND RECTIFYING IMAGES
# This code uses the matrices in .txt output by calibrate_cameras
# and takes the images from the specified folders,
# it undistorts and rectifies the images
# and finally saves the rectified images in tho other specified folders
import cv2
import re
import numpy as np
import glob


class Matrix:
    def __init__(self, matrix_name):
        self.matrix_name = matrix_name
        self.matrix_str = '{}'.format(self.matrix_name) + '_str'
        self.matrix_file = '{}'.format(self.matrix_name) + '_file.txt'
        self.mint = []

    def convert(self, matrix_str):
        newstr = matrix_str.replace("[", "")
        newstr2 = newstr.replace("]", "")
        newstr3 = newstr2.splitlines()
        newstr4 = []
        for el in newstr3:
            el4 = list(map(float, list(filter(None, re.findall(r"\S*", el)))))
            newstr4.append(el4)
        return(newstr4)


def main():
    M1 = Matrix('M1')
    d1 = Matrix('d1')
    M2 = Matrix('M2')
    d2 = Matrix('d2')
    R = Matrix('R')
    T = Matrix('T')
    E = Matrix('E')
    F = Matrix('F')
    matrix_list = []
    matrix_list.append(M1)
    matrix_list.append(d1)
    matrix_list.append(M2)
    matrix_list.append(d2)
    matrix_list.append(R)
    matrix_list.append(T)
    matrix_list.append(E)
    matrix_list.append(F)

    for matrix in matrix_list:
        matrix.matrix_file = open('{}'.format(matrix.matrix_file), 'r')
        matrix.matrix_str = matrix.matrix_file.read()
        matrix.matrix_file.close()
        matrix.mint = matrix.convert(matrix.matrix_str)
        matrix.mint = np.asarray(matrix.mint)
        print('{}'.format(matrix.matrix_name) + '\n', matrix.mint)

    img_path = 'C:/Users/Jared/Documents/python_scripts/'
    images_right = glob.glob(img_path + 'right_test_1/*.PNG')
    images_left = glob.glob(img_path + 'left_test_1/*.PNG')
    #images_left.sort()
    #images_right.sort()

    for i, fname in enumerate(images_right):
        img2 = cv2.imread(images_right[i])
        img1 = cv2.imread(images_left[i])
        cv2.imshow("Original Left Image", img1)
        cv2.imshow("Original RIght Image", img2)
        cv2.waitKey(100)
        h, w = img1.shape[:2]
        # Rectify the images
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(M1.mint, d1.mint, M2.mint, d2.mint, (w, h), R.mint, T.mint, alpha=-1)
        # Calculate new intrinsic matrix
        newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(M1.mint, d1.mint, (w, h), 1, (w, h))
        newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(M2.mint, d2.mint, (w, h), 1, (w, h))
        # Undistort
        mapx1, mapy1 = cv2.initUndistortRectifyMap(M1.mint, d1.mint, None, newcameramtx1, (w, h), 5)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(M2.mint, d2.mint, None, newcameramtx2, (w, h), 5)
        dst1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
        dst2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
        # Crop the images
        x1, y1, w1, h1 = roi1
        x2, y2, w2, h2 = roi2
        dst1 = dst1[y1:y1 + h, x1:x1 + w]
        dst2 = dst2[y2:y2 + h, x2:x2 + w]
        cv2.imwrite(img_path + 'test1_rectified_images_left/' + 'rectified_left_' + '{}'.format(i) + '.png', dst1)
        cv2.imwrite(img_path + 'test1_rectified_images_right/' + 'rectified_right_' + '{}'.format(i) + '.png', dst2)
        cv2.imshow('Left Rectified', dst1)
        cv2.imshow('Right Rectified', dst2)
    print('The rectified and undistorted images have been saved succesfully!')
    cv2.waitKey(0)


main()
