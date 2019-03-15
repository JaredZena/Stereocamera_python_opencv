import cv2
import numpy as np

# Define checkboard size
cols = 9
rows = 7
checkboard_size = (cols,rows)
camL = cv2.VideoCapture(1)
camR = cv2.VideoCapture(2)

img_counter = 0

while True:
    retL_r, frameL = camL.read()
    retR_r, frameR = camR.read()

    #Code to detect checkboard
    gray_frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gray_frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    retL_f, cornersL = cv2.findChessboardCorners(gray_frameL, checkboard_size,
                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH+
                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR_f, cornersR = cv2.findChessboardCorners(gray_frameR, checkboard_size,
                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH+
                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    cv2.imshow("Left Test with chessboard",cv2.drawChessboardCorners(frameL,
                                                                     checkboard_size,cornersL, retL_f))
    cv2.imshow("Right Test with chessboard",cv2.drawChessboardCorners(frameR,
                                                                      checkboard_size,cornersR, retR_f))

    if not retL_r:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32 and retL_f is True and retR_f is True:
        # SPACE pressed
        img_nameL = "opencv_Left_frame_{}.png".format(img_counter)
        img_nameR = "opencv_Right_frame_{}.png".format(img_counter)
        cv2.imwrite(img_nameL, gray_frameL)
        cv2.imwrite(img_nameR, gray_frameR)
        print("New images have been captured and saved!"+"{}".format(img_counter))
        img_counter += 1

camL.release()
camR.release()
cv2.destroyAllWindows()
