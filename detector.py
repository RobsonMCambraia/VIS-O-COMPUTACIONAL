import cv2
import mediapipe as mp
import numpy as np
import time

capture = cv2.VideoCapture("src\hand_01.mp4")
width, height = 740, 580

while True:
    ret, img = capture.read()
    
    img = cv2.resize(img, (width, height))
    
    cv2.imshow("hands", img)
    
    if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
        break
