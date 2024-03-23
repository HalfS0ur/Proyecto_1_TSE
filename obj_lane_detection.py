import cv2
import numpy as np

import json
from openvino.inference_engine import IECore

from common import prepare_image
from common import draw_bounding_boxes

model_name = "ssd_mobilenet_v2_coco"
model_precision = "FP16"

model_xml_path = f"models/ssd_mobilenet_v2_coco/FP16/ssd_mobilenet_v2_coco.xml"
model_bin_path = f"models/ssd_mobilenet_v2_coco/FP16/ssd_mobilenet_v2_coco.bin"

ie = IECore()
# Read network
network = ie.read_network(model_xml_path, model_bin_path)

# Find input shape, layout and size
input_name = next(iter(network.input_info))
input_data = network.input_info[input_name].input_data
input_shape = input_data.shape # [1, 3, 300, 300]
input_layout = input_data.layout # NCHW
input_size = (input_shape[2], input_shape[3]) # (300, 300)

# Load network
device = "CPU" # MYRIAD / CPU
exec_network = ie.load_network(network=network, device_name=device, num_requests=1)

# Load classes
classes_path = f"models/ssd_mobilenet_v2_coco/classes.json"
with open(classes_path) as f:
    classes = f.read()
    
classes = json.loads(classes)


def formato_video(cuadro):
    nuevo_tamanno = cv2.resize(cuadro, (700, 500))
    return nuevo_tamanno

def preprocesamiento(cuadro):
    if cuadro is None:
        video.release()
        cv2.destroyAllWindows()
        exit()
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,180]) #lower_white = np.array([0,0,170])
    upper_white = np.array([106,27,255]) #upper_white = np.array([106,27,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(cuadro, cuadro, mask=mask)
    escala_grises = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(escala_grises, (5, 5), 0)
    filtro_canny = cv2.Canny(blur, 50, 150)
    return filtro_canny

def region_de_interes(imagen_canny):
    vertical = imagen_canny.shape[0]
    horizontal = imagen_canny.shape[1]
    mascara = np.zeros_like(imagen_canny)
    triangulo = np.array([[
    (65, vertical),
    (335, 250),
    (675, horizontal),]], np.int32)
    cv2.fillPoly(mascara, triangulo, 255)
    resultante = cv2.bitwise_and(imagen_canny, mascara)
    return resultante

def transformada_hough(area_canny):
    return cv2.HoughLinesP(area_canny, 1.5, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=500) #return cv2.HoughLinesP(area_canny, 1.3, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=300)

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
        return None, None, None
    
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
    return lineas_promediadas, linea_izquierda, linea_derecha

def dibujar_lineas(cuadro, lineas):
    imagen_lineas = np.zeros_like(cuadro)
    x1, y1, x2, y2 = 0, 0, 0, 0  # Add this line
    if lineas is not None:
        for linea in lineas:
            if linea is not None:  # Check if linea is not None
                for coords in linea:
                    if coords is not None:  # Check if coords is not None
                        # Unpack coordinates
                        x1, y1, x2, y2 = coords

                        # Check if any coordinate is None
                        #if None in (x1, y1, x2, y2):
                            #return imagen_lineas  # Skip drawing the line
                            #print ('none')
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Draw the line
                        cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return imagen_lineas, x1, y1, x2, y2

#Main que no se llama main
video = cv2.VideoCapture("video-calle.mp4")
while(video.isOpened()):
    _, frame = video.read()

    frame_prepared = prepare_image(frame, target_size=input_size, target_layout=input_layout)

    output = exec_network.infer({input_name: frame_prepared})
    detections = output["DetectionOutput"]

    #cv2.imshow('_window_name', draw_bounding_boxes(frame, detections, classes))





    estandarizar_tamanno = formato_video(frame)
    preprocesar_imagen = preprocesamiento(estandarizar_tamanno)
    area_canny = region_de_interes(preprocesar_imagen) #area_canny = region_de_interes(preprocesar_imagen)

    lineas = transformada_hough(area_canny)
    promedio_lineas, linea_i, linea_d = pendiente_promedio(estandarizar_tamanno, lineas)
    imagen_lineas, x1, y1, x2, y2 = dibujar_lineas(estandarizar_tamanno, promedio_lineas)
    playback = cv2.addWeighted(estandarizar_tamanno, 0.8, imagen_lineas, 1, 1)
    end = draw_bounding_boxes(playback, detections, classes)
    #######################
    edge_frame_bgr = cv2.cvtColor(area_canny, cv2.COLOR_GRAY2BGR)
    edge_frame_bgr_resized = cv2.resize(edge_frame_bgr, (estandarizar_tamanno.shape[1], estandarizar_tamanno.shape[0]))

    cuadro_combinado = cv2.hconcat([playback, edge_frame_bgr_resized]) 
    ##############################
    #print(linea_i[0][0], linea_d[0][0]) #quitar
    cv2.imshow("result", end)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
