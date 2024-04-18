import cv2

# Función para detectar bordillos y escaleras en una imagen
def detect_obstacles(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de borde para resaltar los cambios de intensidad
    edges = cv2.Canny(gray, 50, 150)
    
    # Encontrar contornos en la imagen de bordes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Buscar contornos que puedan ser bordillos o escaleras
    obstacle_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filtrar contornos pequeños
            obstacle_detected = True
            # Dibujar un rectángulo alrededor del obstáculo detectado
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return image, obstacle_detected

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar obstáculos en el frame actual
    processed_frame, obstacle_detected = detect_obstacles(frame)
    
    # Mostrar el frame procesado
    cv2.imshow('Obstacle Detection', processed_frame)
    
    # Mostrar una alerta si se detecta un obstáculo
    if obstacle_detected:
        print("¡Se detectó un obstáculo!")
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
