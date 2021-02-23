# Conteo e identificacion de tráfico
Contiene los archivos del proyecto de grado de pregrado de "Conteo e Identificación de tráfico en vídeo mediante un paradigma de inteligencia computacional" 

## Estudiantes
- Sebastián Leonardo Bejarano Medellín
- William Eduardo Sierra Lozano

Integrantes del grupo LASER de la Universidad Distrital Francisco José de Caldas

# Métodos utilizados
- Redes FeedForward (Backpropagation y gradient descent)
- Redes convolucionales (Backpropagation y gradient descent)
- Evolución de clasificadores mediante algoritmo genético
- Transfer Learning

# Pasos del proyecto

## Entrenamiento y obtención de redes neuronales.
### Para cada característica (hasta el momento si es vehículo o no, tipo de vehículo, tipo de matrícula) se deberá obtener cada una de estas redes
1. Obtención de los datasets cuyos links se encontrarán en el archivo datasets.txt TERMINADO
2. Entrenamiento y test de las redes FeedForward utilizando el método de Bag of Visual Words para obtener las características relevantes TERMINADO
3. Entrenamiento y test de las redes convolucionales TERMINADO
4. Evolución y entrenamiento mediante un algoritmo genético simple de las redes convolucionales. TERMINADO
5. Transfer learning con una red convolucional. TERMINADO

## Obtención de las ROI y GUI del algoritmo
1. Implementación con ayuda de OpenCV de los algoritmos de MOG2 y filtrado para obtener la ROI para pasarle a cada red. TERMINADO
2. Implementación de la mejor red obtenida para FF, Convolucional, Evolutiva y Transfer learning para observar los resultados. TERMINADO
3. Tracking de cada objeto para evitar hacer conteo del mismo auto más de una vez TERMINADO
4. Implementación en un sistema embebido con cámara (Google Coral dev Board con TPU) (En proceso)

## Trabajos futuros que pueden servir
1. Sistema de gestión de energía para el dispositivo para que funcione con batería recargable de manera autónoma.
2. Llevar al paradigma de internet de las cosas para tener más de un dispositivo por intersección o donde se quieran colocar.
