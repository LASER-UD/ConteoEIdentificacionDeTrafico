#Importamos las librerías
import tensorflow as tf
import cv2
import dlib
import tensorflow.keras
from tensorflow.keras.models import load_model
import numpy as np

#Rutas importantes
rutaVideo = './Videos_test/test3.mp4'
rutaRed = './Resultados/Convolucional/V_NoV/Arquitectura4_LR_0_01/arq4_1.hdf5'

#clase trackedVehicle
class trackedVehicle:
    trackedVehicles = []    
    def __init__(self, x, y, w, h, vehicleScore, typeScore):
        self.x = x
        self.y = y
        self.vehicleScore = vehicleScore
        self.typeScore = typeScore
        self.centroide = np.array((x+(w/2), y+(h/2)))
        self.tracker = dlib.correlation_tracker()
        self.rect = dlib.rectangle(x,y,x+w,y+h)        
    def nuevoVehiculo(vehiculo):
        noExiste = True
        for i in trackedVehicle.trackedVehicles:
            distancia = np.linalg.norm(i.centroide-vehiculo.centroide)
            if(distancia < 200):
                noExiste = False
        if(noExiste):
            trackedVehicle.trackedVehicles.append(vehiculo)


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
    width = int(imagen.shape[1] * scale_percent / 100)
    height = int(imagen.shape[0] * scale_percent / 100)
    dim = (width, height)
    #resize image
    frame = cv2.resize(imagen, dim, interpolation = cv2.INTER_AREA)
    return frame

def clasificarROI(ROI):
    ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2RGB)
    ROI = cv2.resize(ROI, (200,200), interpolation = cv2.INTER_AREA)
    ROI = ROI.reshape(-1, 200, 200, 3)
    ROI = ROI / 255
    pred = red.predict(ROI)
    return pred[0]

#Creamos elemento de video
vid = cv2.VideoCapture(rutaVideo)

#Creamos el substractor de fondo
mask_fondo = cv2.createBackgroundSubtractorMOG2()

#Cargamos el modelo de Keras
red = load_model(rutaRed)

#Iniciamos un contador de cuadros pasados
frame_count = 0

#Colocamos un contador de saltar cuadros para ahorrar recursos en la detección
skip_frames = 15

#limite de conteo
limite = 0.8

while True:
    ret, frame = vid.read()
    #frame = frame[300:,:,:]
    #Cuadro siguiente
    if ret:        
        #Resize
        resized = escalarImagen(frame,100)
        frame_rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB);
        #Mostramos el vídeo original
        cv2.imshow('frame',resized)
        #Aplicamos el sustractor de fondo
        mask = mask_fondo.apply(resized)
        #Mostramos la máscara
        cv2.imshow('mascara',mask)
        #Filtramos la máscara y la mostramos
        filtrada = filter_mask(mask)
        cv2.imshow('filtrada',filtrada)        
        #creamos una copia del frame original
        frame_copia = resized.copy()
        frame_count = frame_count + 1
        if(frame_count > 200 and frame_count % skip_frames == 0):
            #Ahora detectamos los contornos en la imagen
            contours, hierarchy = cv2.findContours(filtrada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #Dibujamos los bounding boxes para cada contorno
            for c in contours:
                rect = cv2.boundingRect(c)
                x,y,w,h = rect
                if(w > 20 and h > 20):
                    #Pasamos la imagen por el modelo de Keras para ver si es carro o no
                    ROI = frame[y:y+h,x:x+w,:]                             
                    prediccion = clasificarROI(ROI)
                    if(prediccion > 0.4):
                        vehiculo = trackedVehicle(x,y,w,h,prediccion,0)
                        vehiculo.tracker.start_track(frame_rgb,vehiculo.rect)
                        trackedVehicle.nuevoVehiculo(vehiculo)

        
        else:
            for i in trackedVehicle.trackedVehicles:
                i.tracker.update(frame_rgb)
                pos = i.tracker.get_position()                
                #Desempaquetamos la posición
                startX = pos.left()
                startY = pos.top()
                endX = pos.right()
                endY = pos.bottom()
                #actualizamos el centroide
                i.centroide = np.array(((startX+endX)/2,(startY+endY)/2))
                #Ahora sí pasamos todo a enteros para dibujar
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                if(i.centroide[1]>resized.shape[0]*limite):
                    trackedVehicle.trackedVehicles.remove(i)
                # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame_copia, (startX, startY), (endX, endY),(0, 255, 0), 2)	
                cv2.putText(frame_copia,"Vehiculo: "+str(i.vehicleScore),(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1)		    
                image = cv2.circle(frame_copia, (int(i.centroide[0]),int(i.centroide[1])), radius=3, color=(0, 255, 0), thickness=-1)
        #Ahora mostramos el frame copia con los contornos dibujados
        cv2.imshow('BBoxes',frame_copia)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

