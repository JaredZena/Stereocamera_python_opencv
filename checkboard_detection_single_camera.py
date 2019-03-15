import cv2
import numpy as np

# Define checkboard size
cols = 9
rows = 6

cam = cv2.VideoCapture(0)

cv2.namedWindow("test with chessboard")

img_counter = 0

while True:
    ret_r, frame = cam.read()

    #Code to detect checkboard
    checkboard_size = (cols,rows)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_f, corners = cv2.findChessboardCorners(gray_frame, checkboard_size,
                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH+
                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_f is True:
       ret_f = cv2.drawChessboardCorners(frame, checkboard_size,
                                                corners, ret_f)
    
    cv2.imshow("Test with chessboard",frame)
    if not ret_r:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("New images have been captured and saved!"+"{}".format(img_counter))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
