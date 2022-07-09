import cv2
import os
import numpy as np

class Mask_detection:
    def entrenarmascara(self):
        dataPath ="Dataset_faces" # es dinde esta precente todos los datos 
        dir_list = os.listdir(dataPath) 
        print("Lista archivos: ",dir_list) # alistamos la lista de los directorios donde estan las imagenes

        labels =[]  #Son etiquetas asociadas a cada imagen
        facesData=[] #almacenan los rostros
        label = 0 #incrementa conforme a las etiquetas usadas

        for name_dir in dir_list:
            dir_path = dataPath + "/" + name_dir # obtenemos el archivo de con mascarilla y sin mascarilla

            for file_name in os.listdir(dir_path):
                image_path = dir_path + "/" + file_name #obtenemos la ubicacion de los archivos de la imagen 
                print(image_path)
                image = cv2.imread(image_path,0)
                ## como estamos obteniendo correctamente todas las ubicaciones ahora es momento de guardarlos
                facesData.append(image)
                labels.append(label)
            label+=1 #los con mascarilla 0 y sin mascarilla 1

        print("Etiqueta 0: ", np.count_nonzero(np.array(labels)== 0))
        print("Etiqueta 1: ", np.count_nonzero(np.array(labels)== 1))

        # escogeremos (LBPH) local binary patern histogram para el desarrollo 
        # Estamos inicializamos un agente de reconocimiento

        face_mask = cv2.face.LBPHFaceRecognizer_create()

        # ahora lo entrenamos

        print("Entrenando")
        face_mask.train(facesData,np.array(labels))

        #arhora se almacena el modelo

        face_mask.write("face_mask_model.xml")
        print("Almacenado correctamente")
