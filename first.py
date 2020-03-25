import numpy as np
import cv2
cam = cv2.VideoCapture(0)
while True:

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', frame)
    cv2.imshow('Gray', gray)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()