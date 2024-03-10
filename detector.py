import cv2
import mediapipe as mp
import numpy as np
import time
from keras.models import load_model  # TensorFlow is required for Keras to work

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
        self.result = self.hands.process(rgb)

        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                if draw_hands:
                    self.pipe_draw.draw_landmarks(
                        img, hand_landmarks, self.pipe_hands.HAND_CONNECTIONS)
        return img
    
    def encontrar_dedos(self,
                        img: np.ndarray,
                        num_hand: int = 0,):
        self.requerid_landmark_list = []
        
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[0]
            
            for id, landmark in enumerate(my_hand.landmark):
                height, width, _ = img.shape
                center_x = int(landmark.x*width)
                center_y = int(landmark.y*height)
                
                self.requerid_landmark_list.append([id, center_x, center_y])
                
        return self.requerid_landmark_list
    
    def desenhar_info_dedo(self, 
                           img: np.ndarray,
                           info_dedo):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cor_texto = (255, 255, 255)
        tamanho_fonte = 0.5
        espessura_linha = 1

        dedo, posicao_x, posicao_y = map(int, info_dedo)

        texto = f'Dedo: {dedo}, X: {posicao_x}, Y: {posicao_y}'
        cv2.putText(img, texto, (posicao_x, posicao_y - 10), fonte, tamanho_fonte, cor_texto, espessura_linha, cv2.LINE_AA)
        
    def desenhar_box(self, img: np.ndarray):
        height, width, _ = img.shape  #posição da mão

        self.x_max, self.y_max = 0, 0
        self.x_min, self.y_min = width, height

        
        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                self.x_max, self.y_max = float('-inf'), float('-inf')
                self.x_min, self.y_min = float('inf'), float('inf')

                for lm in hand.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    self.x_max = x if x > self.x_max else self.x_max
                    self.x_min = x if x < self.x_min else self.x_min
                    self.y_max = y if y > self.y_max else self.y_max
                    self.y_min = y if y < self.y_min else self.y_min

                return cv2.rectangle(img, (self.x_min - 50, self.y_min - 50), (self.x_max + 50, self.y_max + 50), (0, 255, 0), 2)

    def deteccao_modelo(self, img: np.ndarray, data, model, classes):
        try:
            # Corta a mão da imagem
            img_corte = img[self.y_min - 50: self.y_max + 50, self.x_min - 50: self.x_max + 50]
            
            # Redimensiona para o tamanho esperado pelo modelo
            img_corte = cv2.resize(img_corte, (224, 224))
            
            # Normaliza a imagem
            normalized = (img_corte.astype(np.float32) / 255.0)
            
            # Adiciona uma dimensão extra para criar um lote
            data[0] = np.expand_dims(normalized, axis=0)
            
            # Realiza a predição
            prediction = model.predict(data)
            
            # Obtém o índice da classe prevista
            indexVal = np.argmax(prediction)
            
            # Adiciona a classe prevista à imagem
            cv2.putText(img, classes[indexVal], (self.x_min - 50, self.y_min - 65), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
            
            return img
        
        except Exception as e:
            return (f"Erro na detecção do modelo: {e}")

