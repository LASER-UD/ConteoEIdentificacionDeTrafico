#Importamos las librerías
import tensorflow as tf
import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from collections import OrderedDict
from scipy.spatial import distance as dist

#Rutas importantes
rutaVideo = './Videos_test/test4.mp4'
rutaRedVnV = './modelo_VnV.tflite'
rutaRedTipo = './modelo_Tipo_Transfer.tflite'


#Cargamos el modelo Vehiculo No Vehiculo de TFLite
interprete_VnV = tf.lite.Interpreter(model_path=rutaRedVnV)
interprete_VnV.allocate_tensors()

#Vemos los tensores de entrada y salida
dimension_VnV = 200
input_details_VnV = interprete_VnV.get_input_details()
output_details_VnV = interprete_VnV.get_output_details()


#Cargamos el modelo Tipo de TFLite
interprete_Tipo = tf.lite.Interpreter(model_path=rutaRedTipo)
interprete_Tipo.allocate_tensors()

#Vemos los tensores de entrada y salida
dimension_Tipo = 224
input_details_Tipo = interprete_Tipo.get_input_details()
output_details_Tipo = interprete_Tipo.get_output_details()

class ColorLabeler:

    def labelFunc(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        mean = cv2.mean(image)[:3]
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
		# loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)
        
        return self.colorNames[minDist[1]]
        
    def __init__(self):
		# initialize the colors dictionary, containing the color
		# name as the key and the RGB tuple as the value
        colors = OrderedDict({
			"Rojo": (255, 0, 0),            
			"Verde": (0, 255, 0),            
			"Azul": (0, 0, 255),            
            "Negro": (0, 0, 0),
            "Blanco": (255, 255, 255),            
            "Amarillo" : (236, 208, 0)})
		# allocate memory for the L*a*b* image, then initialize
		# the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
		# loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
		# convert the L*a*b* array from the RGB color space
		# to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    

#clase trackedVehicle
class trackedVehicle:
    trackedVehicles = []    
    def __init__(self, x, y, w, h, vehicleScore, type, typeScore, color):
        self.x = x
        self.y = y
        self.vehicleScore = vehicleScore
        self.type = type
        self.typeScore = typeScore
        self.centroide = np.array((x+(w/2), y+(h/2)))
        self.tracker = dlib.correlation_tracker()
        self.rect = dlib.rectangle(x,y,x+w,y+h)
        self.tiempoLinea = 0.0
        self.velocidad = 0.0   
        self.pasoLinea = False   
        self.color = color
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
    dilation = cv2.dilate(opening, kernel, iterations = 2)

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

def distinguirROI(ROI_1):
    ROI_1 = cv2.cvtColor(ROI_1,cv2.COLOR_BGR2RGB)
    ROI_1 = cv2.resize(ROI_1, (dimension_VnV,dimension_VnV), interpolation = cv2.INTER_AREA)
    ROI_1 = ROI_1.reshape(-1, dimension_VnV, dimension_VnV, 3)
    ROI_1 = np.float32(ROI_1 / 255.0)
    interprete_VnV.set_tensor(input_details_VnV[0]['index'], ROI_1)
    interprete_VnV.invoke()
    pred = interprete_VnV.get_tensor(output_details_VnV[0]['index'])
    return pred[0]

def clasificarVehiculo(ROI_2):
    ROI_2 = cv2.cvtColor(ROI_2,cv2.COLOR_BGR2RGB)
    ROI_2 = cv2.resize(ROI_2, (dimension_Tipo,dimension_Tipo), interpolation = cv2.INTER_AREA)
    ROI_2 = ROI_2.reshape(-1, dimension_Tipo, dimension_Tipo, 3)
    ROI_2 = np.float32(ROI_2 / 1.0)
    interprete_Tipo.set_tensor(input_details_Tipo[0]['index'], ROI_2)
    interprete_Tipo.invoke()
    pred = interprete_Tipo.get_tensor(output_details_Tipo[0]['index'])
    return pred[0]

##Conexión a la base de datos
from pymongo import MongoClient
cliente = MongoClient("mongodb://localhost:27017")
db = cliente.Vehiculos
#Fin conexión a la base de datos 
#Creamos elemento de video
vid = cv2.VideoCapture(rutaVideo)

#Creamos el substractor de fondo
mask_fondo = cv2.createBackgroundSubtractorMOG2()

#Iniciamos un contador de cuadros pasados
frame_count = 0

#Colocamos un contador de saltar cuadros para ahorrar recursos en la detección
skip_frames = 10

#limite de conteo
limite = 0.7
#bandera de carros bajando
Bajando = True

#Porcentaje de tamaño de la ventana boxes
boxesPercent = 70

#Porcentaje de reducción o ampliación de la imagen original
percentFrame = 100

#Vehiculos contados
VehiculosContados = 0

#Linea de la velocidad
if(Bajando):
    lineaVelocidad = limite - 0.1
else:
    lineaVelocidad = 1 -(limite-0.1)
distanciaVelocidad = 10

#Etiquetador de color
cl = ColorLabeler()
while True:
    ret, frame = vid.read()
    frame = frame[100:,650:,:]
    #Cuadro siguiente
    if ret:    
        #INTERFAZ DEL CONTEO DE VEHICULOS        
        #Resize
        resized = escalarImagen(frame,percentFrame)        
        if(Bajando):
            X_start_GUI = 0 
            X_end_GUI = int(resized.shape[1])
            Y_start_GUI = int(limite*resized.shape[0])
            Y_end_GUI = int(resized.shape[0])  
        else:            
            X_start_GUI = 0 
            X_end_GUI = int(resized.shape[1])
            Y_start_GUI = 0
            Y_end_GUI = int((1-limite)*resized.shape[0])        
        frame_rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
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
        cv2.rectangle(frame_copia,(X_start_GUI,Y_start_GUI),(X_end_GUI,Y_end_GUI),(0,0,255),-1)
        cv2.putText(frame_copia,"Vehiculos contados: "+str(VehiculosContados),(int((X_end_GUI+X_start_GUI)/2 - 200),int((Y_end_GUI+Y_start_GUI)/2)),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 0),2)		                    
        cv2.putText(frame_copia,"Linea de velocidad",(0+20,int(resized.shape[0]*lineaVelocidad)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),1)		                    
        cv2.line(frame_copia,(0,int(resized.shape[0]*lineaVelocidad)),(resized.shape[1],int(resized.shape[0]*lineaVelocidad)),(255,0,0),1)
        frame_count = frame_count + 1
        if(frame_count > 100 and frame_count % skip_frames == 0):
            #Ahora detectamos los contornos en la imagen
            if Bajando:
                contours, hierarchy = cv2.findContours(filtrada[:int(limite*filtrada.shape[0]),:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(filtrada[int((1-limite)*filtrada.shape[0]):,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #Dibujamos los bounding boxes para cada contorno
            for c in contours:
                rect = cv2.boundingRect(c)
                x,y,w,h = rect
                if(w > 40 and h > 40):
                    #Pasamos la imagen por el modelo de Keras para ver si es carro o no
                    ROI = frame[y:y+h,x:x+w,:]                                 
                    prediccion = distinguirROI(ROI)                                        
                    if(prediccion > 0.5):
                        pred_tipo = clasificarVehiculo(ROI)                        
                        color = cl.labelFunc(ROI)
                        tipo = np.argmax(pred_tipo)
                        vehiculo = trackedVehicle(x,y,w,h,prediccion,tipo,pred_tipo[tipo], color)
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
                w = endX - startX
                h = endY - startY
                if(w > resized.shape[1]*0.4 or h > resized.shape[0]*0.4):
                    trackedVehicle.trackedVehicles.remove(i)
                if(Bajando):
                    if(i.centroide[1] > resized.shape[0]*lineaVelocidad and i.pasoLinea == False):
                        i.pasoLinea = True
                        i.tiempoLinea = time.time()

                    if(i.centroide[1]>resized.shape[0]*limite):
                        trackedVehicle.trackedVehicles.remove(i)                        
                        VehiculosContados+=1
                        fecha = datetime.now()
                        i.velocidad = 3.6*(distanciaVelocidad / (time.time() - i.tiempoLinea))
                        dict = {
                            "score" : str(i.vehicleScore),
                            "tipo" : str(i.type),
                            "color" : str(i.color),
                            "velocidad" : str(i.velocidad),
                            "hora" : str(fecha.hour)+":"+str(fecha.minute)+":"+str(fecha.second),
                            "fecha" : str(fecha.day)+"-"+str(fecha.month)+"-"+str(fecha.year)                            
                        }
                        result = db.Registro.insert_one(dict)
                else:
                    if(i.centroide[1] < resized.shape[0]*(1-lineaVelocidad) and i.pasoLinea == False):
                        i.pasoLinea = True
                        i.tiempoLinea = time.time()
                    if(i.centroide[1]<resized.shape[0]*(1-limite)):
                        trackedVehicle.trackedVehicles.remove(i)
                        VehiculosContados+=1
                # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame_copia, (startX, startY), (endX, endY),(0, 255, 0), 2)	                
                if(i.type == 0):
                    Tipo = "Ambulancia"
                elif(i.type == 1):
                    Tipo = "Bicicleta"
                elif(i.type == 2):
                    Tipo = "Bus"
                elif(i.type == 3):
                    Tipo = "Camion"
                elif(i.type == 4):
                    Tipo = "Carro"
                elif(i.type == 5):
                    Tipo = "Moto"
                elif(i.type == 6):
                    Tipo = "Taxi"
                elif(i.type == 7):
                    Tipo = "Van"
                cv2.putText(frame_copia,Tipo+"%.2f"%i.typeScore,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0),1)		                    
                cv2.rectangle(frame_copia,(X_start_GUI,Y_start_GUI),(X_end_GUI,Y_end_GUI),(0,0,255),-1)
                cv2.line(frame_copia,(0,int(resized.shape[0]*lineaVelocidad)),(resized.shape[1],int(resized.shape[0]*lineaVelocidad)),(255,0,0),1)
                cv2.putText(frame_copia,"Linea de velocidad",(0+20,int(resized.shape[0]*lineaVelocidad)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),1)		                    
                cv2.putText(frame_copia,"Vehiculos contados: "+str(VehiculosContados),(int((X_end_GUI+X_start_GUI)/2-200),int((Y_end_GUI+Y_start_GUI)/2)),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 0),2)		                    
        #Ahora mostramos el frame copia con los contornos dibujados
        boxesResized = escalarImagen(frame_copia,boxesPercent)
        cv2.imshow('BBoxes',boxesResized)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

