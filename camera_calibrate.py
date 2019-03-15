import numpy as np
import cv2
import glob
import argparse
cols = 9
rows = 7
checkboard_size = (cols, rows)


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((cols * rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        #self.cal_path = "C:\\Users\\Jared\\Documents\\python_scripts\\"
        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        img_shape = (640, 480)
        images_right = glob.glob(cal_path + 'RIGHT/*.PNG')
        images_left = glob.glob(cal_path + 'LEFT/*.PNG')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkboard_size,
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                         cv2.CALIB_CB_FILTER_QUADS)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkboard_size,
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                         cv2.CALIB_CB_FILTER_QUADS)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, checkboard_size, corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(100)

            if ret_r is True:
                rt_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, checkboard_size, corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(800)
            #img_shape = gray_l.shape[::-1]
            # print(img_shape)

        rt_l, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt_r, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims, criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        Intrinsic_mtx_1_file = open("Intrinsic_mtx_1_file.txt", "w")
        dist_1_file = open("dist_1_file.txt", "w")
        Intrinsic_mtx_2_file = open("Intrinsic_mtx_2_file.txt", "w")
        dist_2_file = open("dist_2_file.txt", "w")
        R_file = open("R_file.txt", "w")
        T_file = open("T_file.txt", "w")
        E_file = open("E_file.txt", "w")
        F_file = open("F_file.txt", "w")

        Intrinsic_mtx_1_file.write(str(M1))
        dist_1_file.write(str(d1))
        Intrinsic_mtx_2_file.write(str(M2))
        dist_2_file.write(str(d2))
        R_file.write(str(R))
        T_file.write(str(T))
        E_file.write(str(E))
        F_file.write(str(F))
        #file.write('Camera_model' + '\n' + str(camera_model)+ "\n")

        Intrinsic_mtx_1_file.close()
        dist_1_file.close()
        Intrinsic_mtx_2_file.close()
        dist_2_file.close()
        R_file.close()
        T_file.close()
        E_file.close()
        F_file.close()

        cv2.destroyAllWindows()
        return camera_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
