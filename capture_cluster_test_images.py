## Capture tomato clusters 

import cv2
import numpy as np

# Define checkboard size
camL = cv2.VideoCapture(1)
camR = cv2.VideoCapture(2)

img_counter = 0

while True:
    retL_r, frameL = camL.read()
    retR_r, frameR = camR.read()

    cv2.imshow("Left frame",frameL)
    cv2.imshow("Right frame",frameR)

    if not retL_r:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_nameL = "test_tomatoes_left_{}.png".format(img_counter)
        img_nameR = "test tomatoes_right{}.png".format(img_counter)
        cv2.imwrite(img_nameL, frameL)
        cv2.imwrite(img_nameR, frameR)
        print("New images have been captured and saved!"+"{}".format(img_counter))
        img_counter += 1

camL.release()
camR.release()
cv2.destroyAllWindows()
