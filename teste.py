import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import streamlit as st
from detector import HandsDetector 

st.title("Tradutor de Libras utilizando Visão Computacional")
capture = cv2.VideoCapture(0)
Detector = HandsDetector()
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
model = load_model("model\keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)  # array de entrada

capturing = st.button("Iniciar Captura e Processamento")
image_placeholder = st.empty()


while capturing:
    ret, img = capture.read()
    
    img_processed = Detector.mp_hands(img, draw_hands=True)
    st.image(img_processed, channels="BGR", use_column_width=True, caption="Imagem Processada")
    
    landmark_hand = Detector.encontrar_dedos(img)
    Detector.desenhar_box(img)
    Detector.deteccao_modelo(img, data, model, classes)

    if cv2.waitKey(20) & 0xFF == ord('q') or not ret:
        break

# Libera os recursos quando terminar
capture.release()
cv2.destroyAllWindows()
