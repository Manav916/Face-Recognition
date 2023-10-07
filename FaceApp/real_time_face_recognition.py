import numpy as np
import cv2
from machinelearning import pipeline

cap = cv2.VideoCapture(1)


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    image,res = pipeline(frame)
    
    cv2.imshow('frame',frame)
    cv2.imshow('face recognition',image)
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()