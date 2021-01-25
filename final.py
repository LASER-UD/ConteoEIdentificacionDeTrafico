#Importamos las librerías
import tensorflow as tf
import cv2
import dlib
import tensorflow.keras
from tensorflow.keras.models import load_model

#Rutas importantes
rutaVideo = '../Videos_test/test2.mp4'
rutaRed = '../Resultados/Convolucional/V_NoV/Arquitectura4_LR_0_01/arq4_1.hdf5'

#Función para filtrar la máscara obtenida del background substraction
def filter_mask(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations = 1)

    #Thresholding
    _,thresh1 = cv2.threshold(dilation,254,255,cv2.THRESH_BINARY)
    return thresh1

#Devuelve la imagen escalada a cierto porcentaje consevando la proporción
def escalarImagen(imagen, porcentaje):
    #Resize
    scale_percent = porcentaje # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    #resize image
    frame = cv2.resize(imagen, dim, interpolation = cv2.INTER_AREA)
    return frame