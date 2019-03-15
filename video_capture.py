import numpy as np
import cv2

cap1 = cv2.VideoCapture(2)
#cap2 = cv2.VideoCapture(2)

while(True):
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    #ret2, frame2 = cap2.read()

    # Our operations on the frame come here
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
   # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame1', gray1)
    # cv2.imshow('frame2',gray2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()
