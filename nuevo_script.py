import cv2
import numpy as np

def formato_video(cuadro):
    nuevo_tamanno = cv2.resize(cuadro, (700, 500))
    return nuevo_tamanno

##TODO FILTRAR EL VIDEO POR COLOR, DEJAR SOLO LAS LINEAS BLANCAS y amarillas

def preprocesamiento(cuadro):
    if cuadro is None:
        video.release()
        cv2.destroyAllWindows()
        exit()
    escala_grises = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(escala_grises, (5, 5), 0)
    filtro_canny = cv2.Canny(blur, 50, 150)
    return filtro_canny

def region_de_interes(imagen_canny):
    vertical = imagen_canny.shape[0]
    horizontal = imagen_canny.shape[1]
    mascara = np.zeros_like(imagen_canny)
    triangulo = np.array([[
    (90, vertical),
    (390, 230),
    (600, horizontal),]], np.int32)
    cv2.fillPoly(mascara, triangulo, 255)
    resultante = cv2.bitwise_and(imagen_canny, mascara)
    return resultante

def transformada_hough(area_canny):
    return cv2.HoughLinesP(area_canny, 1.3, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=300)

def calcular_puntos(cuadro, lineas):
    if lineas is None:
        return None
    pendiente, intercepcion = lineas
    y1 = int(cuadro.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercepcion) / pendiente)
    x2 = int((y2 - intercepcion) / pendiente)
    return [[x1, y1, x2, y2]]

def pendiente_promedio(imagen, lineas):
    izquierda = []
    derecha = []
    if lineas is None:
        return None
    for linea in lineas:
        for x1, y1, x2, y2 in linea:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            pendiente = fit[0]
            intercepcion = fit[1]
            if pendiente < 0:
                izquierda.append((pendiente, intercepcion))
            else:
                derecha.append((pendiente, intercepcion))
    promedio_izquierda = np.average(izquierda, axis = 0) if izquierda else None
    promedio_derecha = np.average(derecha, axis = 0) if derecha else None
    linea_izquierda = calcular_puntos(imagen, promedio_izquierda)
    linea_derecha = calcular_puntos(imagen, promedio_derecha)
    lineas_promediadas = [linea_izquierda, linea_derecha]
    return lineas_promediadas

def dibujar_lineas(cuadro, lineas):
    imagen_lineas = np.zeros_like(cuadro)
    if lineas is not None:
        for linea in lineas:
            if linea is not None:  # Check if linea is not None
                for coords in linea:
                    if coords is not None:  # Check if coords is not None
                        # Unpack coordinates
                        x1, y1, x2, y2 = coords

                        # Check if any coordinate is None
                        if None in (x1, y1, x2, y2):
                            return imagen_lineas  # Skip drawing the line
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Draw the line
                        cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return imagen_lineas

#Main que no se llama main
video = cv2.VideoCapture("test2.mp4")
while(video.isOpened()):
    _, frame = video.read()
    estandarizar_tamanno = formato_video(frame)
    preprocesar_imagen = preprocesamiento(estandarizar_tamanno)
    area_canny = region_de_interes(preprocesar_imagen)

    lineas = transformada_hough(area_canny)
    promedio_lineas = pendiente_promedio(estandarizar_tamanno, lineas)
    imagen_lineas = dibujar_lineas(estandarizar_tamanno, promedio_lineas)
    playback = cv2.addWeighted(estandarizar_tamanno, 0.8, imagen_lineas, 1, 1)

    print(pendiente_promedio(estandarizar_tamanno, lineas)) #quitar
    cv2.imshow("result", playback)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
