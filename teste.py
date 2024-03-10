from keras.models import load_model
import cv2
import numpy as np

class HandsDetector:
    def __init__(self, model_path, labels_path):
        # Load the model
        self.model = load_model(model_path, compile=False)

        # Load the labels
        self.class_names = [line.strip() for line in open(labels_path, "r")]

    def deteccao_modelo(self, img):
        try:
            # Resize the raw image into (224-height, 224-width) pixels
            img_corte = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            # Show the image in a window
            cv2.imshow("Webcam Image", img_corte)

            # Make the image a numpy array and reshape it to the model's input shape.
            img_corte = np.asarray(img_corte, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            img_corte = (img_corte / 127.5) - 1

            # Predicts the model
            prediction = self.model.predict(img_corte)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            print("Class:", class_name, end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

            return img

        except Exception as e:
            print(f"Erro na detecção do modelo: {e}")
            return img


# CAMERA pode ser 0 ou 1 com base na câmera padrão do seu computador
camera = cv2.VideoCapture("src\hand_01.mp4")

# Carrega o detector de mãos
detector = HandsDetector("model/keras_model.h5", "model/labels.txt")

while True:
    # Captura a imagem da webcam.
    ret, image = camera.read()

    # Detecta a mão e exibe informações na imagem
    image_with_detection = detector.deteccao_modelo(image)

    # Escuta o teclado para pressionamentos de tecla.
    keyboard_input = cv2.waitKey(1)

    # 27 é o ASCII para a tecla ESC no seu teclado.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
