import cv2
import mediapipe as mp
import numpy as np
import time

capture = cv2.VideoCapture("src\hand_01.mp4")

while True:
    _, img = capture.read()
    cv2.imshow("hands", img)
    
    if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
        break
