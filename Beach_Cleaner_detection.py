import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

def configurar_camara()
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 1280)
    picam2.preview_configuration.main.format = RGB888
    picam2.preview_configuration.align()
    picam2.configure(preview)
    picam2.start()
    return picam2

def detectar_objetos_yolo(model, frame)
    results = model(frame, classes=[0, 1, 2, 3, 4])
    return results

def detectar_mar_azul(frame)
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_azul = np.array([135, 100, 100], dtype=np.uint8)
    upper_azul = np.array([155, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsvImage, lower_azul, upper_azul)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_grandes = []
    for contorno in contornos
        area = cv2.contourArea(contorno)
        if area  10000
            contornos_grandes.append(contorno)
    
    return contornos_grandes

def detectar_cajas_rojas(frame)
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_rojo1 = np.array([0, 150, 150], dtype=np.uint8)
    upper_rojo1 = np.array([10, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsvImage, lower_rojo1, upper_rojo1)
    
    lower_rojo2 = np.array([170, 150, 150], dtype=np.uint8)
    upper_rojo2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsvImage, lower_rojo2, upper_rojo2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contorno in contornos
        area = cv2.contourArea(contorno)
        if area  1000
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        bboxes.append((x, y, w, h))
    
    return bboxes

def calcular_comando(results, contornos_mar, bboxes_rojas, frame_width, frame_height)
    OBJETOS_EVADIR = [chair, person, umbrella, manikin]
    OBJETOS_RECOGER = [can]
    
    if len(contornos_mar)  0
        contorno_mayor = max(contornos_mar, key=cv2.contourArea)
        M = cv2.moments(contorno_mayor)
        if M[m00] != 0
            cx = int(M[m10]  M[m00])
            
            if cx  frame_width  2
                return RIGHT45
            else
                return LEFT45
    
    if len(bboxes_rojas)  0
        x, y, w, h = bboxes_rojas[0]
        cx = x + w2
        area = w  h
        
        if area  50000
            return TURN180
        elif cx  frame_width3
            return LEFT20
        elif cx  2frame_width3
            return RIGHT20
        else
            return FORWARD30
    
    detections = results[0].boxes
    
    for detection in detections
        cls = int(detection.cls[0])
        bbox = detection.xyxy[0].cpu().numpy()
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2)  2
        w = x2 - x1
        h = y2 - y1
        area = w  h
        
        class_name = results[0].names[cls]
        
        if class_name in OBJETOS_EVADIR
            if area  200000
                return BACK50
            elif cx  frame_width3
                return RIGHT30
            elif cx  2frame_width3
                return LEFT30
            else
                return BACK20
        
        elif class_name in OBJETOS_RECOGER
            if area  100000
                return COLLECT
            elif cx  frame_width3
                return LEFT15
            elif cx  2frame_width3
                return RIGHT15
            else
                return FORWARD40
    
    return FORWARD20

def dibujar_bboxes(frame, bboxes, label, color)
    for (x, y, w, h) in bboxes
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def dibujar_fps(frame, inference_time)
    fps = 1000  inference_time
    text = f'FPS {fps.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def main()
    picam2 = configurar_camara()
    model = YOLO(canv1.pt)
   
    
    while True
        frame = picam2.capture_array()
        frame_height, frame_width = frame.shape[2]
        
        results = detectar_objetos_yolo(model, frame)
        annotated_frame = results[0].plot()
        
        bboxes_rojas = detectar_cajas_rojas(frame)
        annotated_frame = dibujar_bboxes(annotated_frame, bboxes_rojas, box, (0, 0, 255))
        
        contornos_mar = detectar_mar_azul(frame)
        cv2.drawContours(annotated_frame, contornos_mar, -1, (255, 0, 0), 3)
        
        comando = calcular_comando(results, contornos_mar, bboxes_rojas, 
                                   frame_width, frame_height)
        
        print(fComando {comando})
      
        cv2.putText(annotated_frame, fCMD {comando}, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        inference_time = results[0].speed['inference']
        annotated_frame = dibujar_fps(annotated_frame, inference_time)
        
        cv2.imshow(Camera, annotated_frame)
        
        if cv2.waitKey(1) == ord(q)
            break
    
    cv2.destroyAllWindows()

if __name__ == __main__
    main()
