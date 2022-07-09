import cv2
import os
import mediapipe as mp
from PIL import Image
import pickle as pc
import numpy as np
from datetime import datetime
from Interfaz import Aplicacion
#librerias para manejar el serial
import serial
import time

class App_mask:
    def Ejecutar(self):
        ser =serial.Serial('COM11',9600,timeout=1)
        time.sleep(2)
        lista = open("lista_usuarios.txt","w")
        # Estaremos usando un detector de media pipe
        mp_face_detection = mp.solutions.face_detection
        now = datetime.now()
        # los labels a obtener son dos
        labels = ["Con_mascarilla", "Sin_mascarilla"]
        # Cargamos el clasificador que es la red neuronal que nos dará la etiqueta
        cascade_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
        # Ahora inicializamos un agente de reconocimiento
        # Ahora cargamos el recognizer con la data previamente entrenada
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('face-training.xml')
        # leeriamos el modelo creado
        face_mask = cv2.face.LBPHFaceRecognizer_create()
        face_mask.read("face_mask_model.xml")
        # Una vez el recognizer tiene la data, recuperamos los labels
        labelssation = {}
        with open('pickles/face-labels.pickle', 'rb') as file:
            # Deserializamos el picle para obtener los labels
            labels_bytes = pc.load(file)
            # Utilizamos list comprenhension para obtener los valores
            labelssation = {value: key for key, value in labels_bytes.items()}
        # empezaremos con la lectura del video
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while True:
                ret, frame = cap.read()
                if ret == False:
                    break
                #frame = cv2.wflip(frame, 1)  # invertimos la camara

                # detectaremos el rostro con mediapipe
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                # Obtenemos la imagen con el filtro de escala de grises
                gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Recorreremos los elementos leidos por el recognizer para determinar las regiones de interes
                face_detectors = cascade_classifier.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=8)
                if results.detections is not None:
                    for detection in results.detections:
                        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = int(detection.location_data.relative_bounding_box.width * width)
                        h = int(detection.location_data.relative_bounding_box.height * height)

                        # si nos detecto el rostro y se marca con un rectangulo
                        # cv2.rectangle(frame,(xmin, ymin),(xmin + w, ymin + h),(0,255,0),5)

                        if xmin < 0 and ymin <0:
                            continue
                        
                        # Obtenemos la region de interes
                        roi = gray_scale[ymin: ymin + height, xmin: xmin + width]
                        # Obtenemos la posicion de la etiqueta a mostrar
                        label_position, accuracy = recognizer.predict(roi)    
                        
                        face_image = frame[ymin:ymin+h,xmin:xmin+w] #obtiene ancho y alto de la cara
                        face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY) #transformamos a escala de grices
                        face_image = cv2.resize(face_image,(72,72),interpolation=cv2.INTER_CUBIC) #redimencionamos a la escala en la que estan todas las imagenes
                        #cv2.imshow("Rostro",face_image) # asi veremos que hace todo

                        # ahora aplicamos el modelo
                        result = face_mask.predict(face_image) # nos da dos valores el lable de la prediccion y el otro es el valor de confianza
                        #cv2.putText(frame,"{}".format(result),(xmin,ymin-5),1,1.3,(210,124,176),1,cv2.LINE_AA)

                        if(result[1]<150):
                            color=(0,255,0) if labels[result[0]] == "Con_mascarilla" else (0,0,255)
                            print(accuracy)
                            tiempo= now.strftime('%d/%m/%Y %H:%M')
                            if accuracy <= 73:
                                name = labelssation[label_position]
                                if labels[result[0]] == "Con_mascarilla":
                                    ser.write(b'P')
                                    lista.write(name + " " + labels[result[0]]+" "+tiempo+"\n")
                                else:
                                    lista.write(name + " " +labels[result[0]]+" "+tiempo+"\n")
                                    ser.write(b'N')
                                # Definimos los valores a dibujar en la imagen
                                cv2.putText(frame, name,(xmin,ymin-5),1,1.3,(210,124,176),1,cv2.LINE_AA)
                            else:
                                cap.release()
                                cv2.destroyAllWindows()
                                ser.close()
                                aplicacion1=Aplicacion()   
                                self.Ejecutar()
                            cv2.putText(frame,"{}".format(labels[result[0]]),(xmin,ymin-25),2,1,color,1,cv2.LINE_AA)#si quieres ver valor de resul pon -25
                            cv2.rectangle(frame,(xmin, ymin),(xmin + w, ymin + h),color,2)
                cv2.imshow("Detector de Barbijos", frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
        ser.close()