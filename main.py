import cv2
import mediapipe as mp
import numpy as np
import time
from detector import HandsDetector

capture = cv2.VideoCapture("src\hand_01.mp4")
# capture = cv2.VideoCapture(0)
width, height = 740, 580

Detector = HandsDetector()

while True:
    ret, img = capture.read()

    img = cv2.resize(img, (width, height))
    Detector.mp_hands(img, draw_hands=True)
    landmark_hand = Detector.encontrar_dedos(img)
    if landmark_hand:
        for dedo_info in landmark_hand:
            dedo, _, _ = dedo_info
            if dedo in (2, 8): 
                Detector.desenhar_info_dedo(img, dedo_info)        

    cv2.imshow("hands", img)

    if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
        break

capture.release()
cv2.destroyAllWindows()