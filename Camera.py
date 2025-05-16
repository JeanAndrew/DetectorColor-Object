import cv2
from ultralytics import YOLO

# Cargar modelo YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Traducción de etiquetas de COCO (en inglés) a español
label_translation = {
    "person": "persona", "bicycle": "bicicleta", "car": "coche", "motorcycle": "motocicleta",
    "airplane": "avión", "bus": "autobús", "train": "tren", "truck": "camión", "boat": "barco",
    "traffic light": "semáforo", "fire hydrant": "hidrante", "stop sign": "señal de stop",
    "parking meter": "parquímetro", "bench": "banco", "bird": "pájaro", "cat": "gato",
    "dog": "perro", "horse": "caballo", "sheep": "oveja", "cow": "vaca", "elephant": "elefante",
    "bear": "oso", "zebra": "cebra", "giraffe": "jirafa", "backpack": "mochila", "umbrella": "paraguas",
    "handbag": "bolso", "tie": "corbata", "suitcase": "maleta", "frisbee": "frisbi", "skis": "esquís",
    "snowboard": "snowboard", "sports ball": "pelota", "kite": "cometa", "baseball bat": "bate",
    "baseball glove": "guante de béisbol", "skateboard": "monopatín", "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis", "bottle": "botella", "wine glass": "copa", "cup": "taza",
    "fork": "tenedor", "knife": "cuchillo", "spoon": "cuchara", "bowl": "cuenco", "banana": "plátano",
    "apple": "manzana", "sandwich": "sándwich", "orange": "naranja", "broccoli": "brócoli",
    "carrot": "zanahoria", "hot dog": "perrito caliente", "pizza": "pizza", "donut": "dónut",
    "cake": "pastel", "chair": "silla", "couch": "sofá", "potted plant": "planta", "bed": "cama",
    "dining table": "mesa de comedor", "toilet": "inodoro", "tv": "televisor", "laptop": "portátil",
    "mouse": "ratón", "remote": "control", "keyboard": "teclado",
    "microwave": "microondas", "oven": "horno", "toaster": "tostadora", "sink": "fregadero",
    "refrigerator": "refrigerador", "book": "libro", "clock": "reloj", "vase": "jarrón",
    "scissors": "tijeras", "teddy bear": "oso de peluche", "hair drier": "secador",
    "llaves": "keys",    "cell phone": "Telefono"
}
# Agregar alias para entrada del usuario (soporte extendido)T
input_aliases = {
    "billetera": "handbag",
    "mochila": "backpack",
    "móvil": "cell phone",
    "persona": "person",
    "botella": "bottle",
    "portátil": "laptop",
    "libro": "book",
    "llaves": "no_detectable"  # para manejarlo como no soportado
}
# Mostrar todas las clases disponibles del modelo
print("🧠 Clases disponibles en el modelo:")
print(", ".join(model.names.values()))

# Pedir al usuario el objeto a detectar
target_input = input("\n📝 Escribe el nombre del objeto que deseas detectar (en inglés o español): ").strip().lower()

# Convertir entrada a inglés si está en español
reverse_translation = {v.lower(): k for k, v in label_translation.items()}
target_object_en = reverse_translation.get(target_input, target_input)  # Usa original si no está traducido

# Verificar si el objeto existe en el modelo
if target_object_en not in model.names.values():
    print(f"❌ El objeto '{target_input}' no está disponible en el modelo YOLO.")
    exit()

# Obtener índice de clase
target_class_id = [k for k, v in model.names.items() if v == target_object_en][0]
target_label_es = label_translation.get(target_object_en, target_object_en)

print(f"🎯 Buscando objeto: '{target_label_es}'\n")

# Color del cuadro
box_color = (0, 255, 0)

# Iniciar cámara
cap = cv2.VideoCapture(0)
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break


    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        paused = not paused
        print("⏸️ Pausado" if paused else "▶️ Reanudado")

    if key == 27:
        print("🚪 Cerrando por tecla ESC...")
        break

    if paused:
        cv2.putText(frame, "⏸️ PAUSADO", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Detección", frame)
        continue

    results = model(frame, verbose=False)[0]

    # Filtrar solo las detecciones del objeto deseado
    boxes = [b for b in results.boxes if int(b.cls[0]) == target_class_id]

    if boxes:
        # Mostrar solo el objeto más grande
        biggest_box = max(
            boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )

        x1, y1, x2, y2 = map(int, biggest_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, target_label_es, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    cv2.imshow("Detección", frame)

# Cerrar
cap.release()
cv2.destroyAllWindows()
