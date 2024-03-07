import cv2
import mediapipe as mp
import numpy as np
import time

class HandsDetector:
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
        self.hands = self.pipe_hands.Hands(self.mode, 
                                         self.n_hands, 
                                         self.complexity, 
                                         self.min_detection, 
                                         self.min_traking)
        
        self.pipe_draw = mp.solutions.drawing_utils

    def mp_hands(self, img: np.ndarray, draw_hands: bool = True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if draw_hands:
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
if __name__ == '__main__':
    capture = cv2.VideoCapture("src/hand_01.mp4")
    width, height = 740, 580

    Detector = HandsDetector()

    while True:
        ret, img = capture.read()

        img = cv2.resize(img, (width, height))
        Detector.mp_hands(img, draw_hands=False)
        cv2.imshow("hands", img)

        if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
            break

    capture.release()
    cv2.destroyAllWindows()