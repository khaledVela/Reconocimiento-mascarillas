import cv2
import os

from PIL import Image
import pickle as pc
import numpy as np


class RecognitionMethods:
    def __init__(self):
        self.IMAGE_DIR = 'images/'
        self.NATIVE_CAMERA = 0
        self.AUXILIAR_CAMERA = 1

    def nuevoUsuario(self, folder_name, photo_qty=1155, camera=0):
        # Inicializar el capturador
        catcher = cv2.VideoCapture(camera)
        # Crear la carpeta contenedora de la imagen utilizando el folder name y el Base Dir
        os.mkdir(self.IMAGE_DIR + folder_name)
        counter = 1
        while(True):
            # Obtenemos el frame para guardar la imagen
            state, frame = catcher.read()
            # Convierto el frame capturado a escala de grises
            gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Ahora procedemos a guardar la imagen
            file_name = f'PIC-{counter}.png'
            cv2.imwrite(self.IMAGE_DIR + folder_name +
                        '/'+file_name, cv2.split(gray_filter)[2])
            # Ahora mostramos la imagen
            cv2.imshow('Obteniendo imagenes...', frame)
            counter += 1
            if counter > photo_qty:
                break
        # Limpiamos la memoria destruyendo las ventanas
        cv2.destroyAllWindows()

    def EntrenarUsuario(self):
        # Primero inicializamos la Red Neuronal Clasificatoria
        cascade_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt2.xml')
        # Ahora inicializamos un agente de reconocimiento
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Preparamos la data presente en la imagen
        current_position = -1
        x_train = []
        y_train = []
        labels = {}  # {'EDUARDOFLORES': 0, 'JUANPEREZ': 1....}
        # Para rellenar estas estructuras, recorreremos el ImageDir y trabajaremos con cada folder y su contenido
        for root, folders, files in os.walk(self.IMAGE_DIR):
            for image in files:
                if image.endswith('jpg') or image.endswith('png') or image.endswith('gif') or image.endswith('jpeg'):
                    # Obtenemos la imagen para procesar
                    path = os.path.join(root, image)
                    label = os.path.basename(root).replace(' ', '').upper()
                    # Si el label no existe en el diccionario, se lo agrega
                    if label not in labels:
                        current_position += 1
                        labels[label] = current_position
                    # Cargar la imagen para la conversión a un arreglo multidimensional numérico
                    image_pil = Image.open(path).convert('L')
                    image_bytes = image_pil.resize((500, 500), Image.ANTIALIAS)
                    # Ahora si obtenemos el numpy array
                    image_np = np.array(image_bytes, 'uint8')
                    # Ahora trabajamos en los datos de pre-entrenamiento con la ayuda del classifier
                    face_detector = cascade_classifier.detectMultiScale(
                        image_np, scaleFactor=1.5, minNeighbors=8)
                    for (x, y, width, height) in face_detector:
                        # Definimos la región de interés
                        roi = image_np[y: y + height, x: x + width]
                        # Una vez tenemos la región de interés, lo guardamos en train_x
                        x_train.append(roi)
                        y_train.append(current_position)
        # Serializamos la data para facilitar su uso en la predicción
        with open('pickles/face-labels.pickle', 'wb') as file:
            pc.dump(labels, file)
        # Ahora guardamos la data de entrenamiento en el folder training
        recognizer.train(x_train, np.array(y_train))
        recognizer.save('face-training.xml')
