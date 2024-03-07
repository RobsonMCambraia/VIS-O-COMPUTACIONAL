import cv2
import mediapipe as mp
import numpy as np
import time

class handsDetector:
    def __init__(self, 
                 mode: bool = False,
                 n_hands: int = 2,
                 complexity: int = 1,
                 min_detection: float = 0.5,
                 min_traking: float = 0.5):
        self.mode = mode
        self.n_hands = n_hands
        self.complexity = complexity
        self.min_detection = min_detection
        self.min_traking = min_traking
        
        self.pipe_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, 
                                         self.max_hands, 
                                         self.complexity_level, 
                                         self.min_detection_conf, 
                                         self.min_traking_conf)
        
        self.pipe_draw = mp.solutions.drawing_utils

if __name__ == '__main__':
    capture = cv2.VideoCapture("src\hand_01.mp4")
    width, height = 740, 580

    while True:
        ret, img = capture.read()
        
        img = cv2.resize(img, (width, height))
        
        cv2.imshow("hands", img)
        
        if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
            break
    capture.release()
    cv2.destroyAllWindows()