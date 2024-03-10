import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from detector import HandsDetector 

# capture = cv2.VideoCapture("src\hand_01.mp4")
capture = cv2.VideoCapture(0)

Detector = HandsDetector()

# Modelo keras
# classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
model = load_model("model\keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)  # array de entrada
# classes = [line.strip() for line in open("model\labels.txt", "r")]

while True:
    ret, img = capture.read()

    Detector.mp_hands(img, draw_hands=True)
    landmark_hand = Detector.encontrar_dedos(img)
    Detector.desenhar_box(img)
    Detector.deteccao_modelo(img, data, model, classes)

    cv2.imshow("hands", img)

    if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
        break

capture.release()
cv2.destroyAllWindows()
