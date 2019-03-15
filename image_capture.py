import cv2

camL = cv2.VideoCapture(0)
camR = cv2.VideoCapture(2)

cv2.namedWindow("test Left")
cv2.namedWindow("test Right")

img_counter = 0

while True:
    retL, frameL = camL.read()
    retR, frameR = camR.read()
    cv2.imshow("test Left", frameL)
    cv2.imshow("test Right", frameR)
    
    if not retL:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_nameL = "opencv_Left_frame_{}.png".format(img_counter)
        img_nameR = "opencv_Right_frame_{}.png".format(img_counter)
        cv2.imwrite(img_nameL, frameL)
        cv2.imwrite(img_nameR, frameR)
        print("New images have been captured and saved!"+"{}".format(img_counter))
        img_counter += 1

camL.release()
camR.release()

cv2.destroyAllWindows()
