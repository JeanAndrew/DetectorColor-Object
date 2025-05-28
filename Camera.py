import cv2
from ultralytics import YOLO
import numpy as np

# Cargar el modelo YOLO
model = YOLO("yolov8n.pt")

# Traducción de etiquetas al español (resumida aquí)
label_translation = {
      "person": "persona", "bicycle": "bicicleta", "car": "coche", "motorcycle": "motocicleta",
    "traffic light": "semáforo", "fire hydrant": "hidrante", "stop sign": "señal de stop",
    "parking meter": "parquímetro", "backpack": "mochila", "umbrella": "paraguas",
    "handbag": "bolso", "tie": "corbata", "suitcase": "maleta", "frisbee": "frisbi", "skis": "esquís",
    "snowboard": "snowboard", "sports ball": "pelota", "kite": "cometa", "baseball bat": "bate",
    "baseball glove": "guante de béisbol", "skateboard": "monopatín", "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis", "bottle": "botella", "wine glass": "copa", "cup": "taza",
    "fork": "tenedor", "knife": "cuchillo", "spoon": "cuchara", "bowl": "cuenco", "banana": "plátano",
    "apple": "manzana", "sandwich": "sándwich", "orange": "naranja", "broccoli": "brócoli",
    "carrot": "zanahoria", "hot dog": "perrito caliente", "pizza": "pizza", "donut": "dona",
    "cake": "pastel", "chair": "silla", "couch": "sofá", "potted plant": "planta", "bed": "cama",
    "dining table": "mesa de comedor", "toilet": "inodoro", "tv": "televisor", "laptop": "portátil",
    "mouse": "ratón", "remote": "control", "keyboard": "teclado", "cell phone": "celular",
    "microwave": "microondas", "oven": "horno", "toaster": "tostadora", "sink": "fregadero",
    "refrigerator": "refrigerador", "book": "libro", "clock": "reloj", "vase": "jarrón",
    "scissors": "tijeras", "teddy bear": "oso de peluche", "hair drier": "secador",
    "toothbrush": "cepillo de dientes", "llaves": "keys" 
}

# Colores conocidos (ejemplo simple)
known_colors = {
        # Rojos
    "rojo": (255, 0, 0),
    "rojo oscuro": (139, 0, 0),
    "rojo claro": (255, 99, 71),
    "rojo sangre": (178, 34, 34),
    "rojo ladrillo": (203, 65, 84),
    "rojo cereza": (220, 20, 60),
    "rojo carmesí": (153, 0, 0),
    "rojo escarlata": (255, 36, 0),

    # Verdes
    "verde": (0, 255, 0),
    "verde oscuro": (0, 100, 0),
    "verde claro": (144, 238, 144),
    "verde lima": (50, 205, 50),
    "verde bosque": (34, 139, 34),
    "verde menta": (152, 255, 152),
    "verde oliva": (128, 128, 0),
    "verde esmeralda": (80, 200, 120),
    "verde jade": (0, 168, 107),

    # Azules
    "azul": (0, 0, 255),
    "azul oscuro": (0, 0, 139),
    "azul claro": (173, 216, 230),
    "azul cielo": (135, 207, 235),
    "azul marino": (0, 0, 128),
    "azul cobalto": (0, 71, 171),
    "azul eléctrico": (11, 0, 255),
    "azul real": (65, 105, 225),
    "azul pizarra": (106, 90, 205),

    # Amarillos
    "amarillo": (255, 255, 0),
    "amarillo claro": (255, 255, 153),
    "amarillo oscuro": (184, 134, 11),
    "amarillo limón": (255, 250, 205),
    "amarillo dorado": (255, 215, 0),
    "amarillo mostaza": (255, 219, 88),

    # Naranjas
    "naranja": (255, 165, 0),
    "naranja oscuro": (255, 140, 0),
    "naranja claro": (255, 222, 173),
    "naranja calabaza": (255, 117, 24),
    "naranja óxido": (184, 115, 51),
    "naranja melocotón": (255, 218, 185),

    # Rosados
    "rosado": (255, 192, 203),
    "rosado claro": (255, 228, 225),
    "rosa fuerte": (255, 20, 147),
    "rosa chicle": (255, 105, 180),
    "rosa salmón": (250, 128, 114),
    "rosa palo": (219, 112, 147),

    # Morados
    "morado": (128, 0, 128),
    "morado oscuro": (75, 0, 130),
    "morado claro": (221, 160, 221),
    "violeta": (148, 0, 211),
    "lavanda": (230, 230, 250),
    "amatista": (153, 102, 204),
    "malva": (224, 176, 255),

    # Cianos y Turquesas
    "cian": (0, 255, 255),
    "cian oscuro": (0, 139, 139),
    "cian claro": (224, 255, 255),
    "turquesa": (64, 224, 208),
    "turquesa oscuro": (0, 206, 209),

    # Blancos y Negros
    "blanco": (150, 100, 140),
    "blanco ahumado": (245, 245, 245),
    "blanco marfil": (255, 255, 240),
    "blanco antiguo": (250, 235, 215),
    "negro": (80, 103, 140), # Aquí el negro puro
    "negro azabache": (25, 25, 25),
    "negro carbón": (50, 50, 50),
    "negro ultra oscuro": (5, 5, 5),
    "negro de la noche": (10, 10, 10),

    # Marrones
    "marrón": (139, 69, 19),
    "marrón oscuro": (101, 67, 33),
    "marrón claro": (220, 160, 222),
    "marrón chocolate": (123, 63, 0),
    "marrón café": (76, 38, 0),
    "marrón rojizo": (165, 42, 42),
    "sepia": (112, 66, 20),
    "terracota": (204, 78, 92),
    "beige": (245, 245, 220),
    "crema": (255, 253, 208),
}

# Función para obtener el color dominante
def get_dominant_color(image, k=1):
    if image.size == 0:
        return 0, 0, 0
    image = cv2.resize(image, (100, 100))
    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    bgr = palette[0].astype(int)
    return bgr[2], bgr[1], bgr[0]  # RGB

# Función para traducir color RGB a nombre
def rgb_to_color_name(r, g, b):
    min_dist = float("inf")
    closest_color = "color indefinido"
    for name, (cr, cg, cb) in known_colors.items():
        dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    return closest_color

# Mostrar clases disponibles
available_classes = [label_translation.get(name.lower(), name) for name in model.names.values()]
print("🧠 Clases disponibles en el modelo:")
print(", ".join(available_classes))

# Entrada del usuario
target_input = input("\n📝 Objeto a detectar (inglés o español): ").strip().lower()
reverse_translation = {v: k for k, v in label_translation.items()}
target_object_en = reverse_translation.get(target_input, target_input)

if target_object_en not in model.names.values():
    print(f"❌ El objeto '{target_input}' no está disponible en el modelo YOLO.")
    exit()

target_class_id = [k for k, v in model.names.items() if v == target_object_en][0]
target_label_es = label_translation.get(target_object_en, target_object_en)
print(f"🔎 Buscando: '{target_label_es}'\n")

# --- CÁMARA: INTENTA CÁMARA DEL MÓVIL PRIMERO ---
ip_cam_url = "http://100.93.241.248:8080/video"
cap = cv2.VideoCapture(ip_cam_url)

# Si falla, usa la webcam local
if not cap.isOpened():
    print("⚠️ No se pudo abrir la cámara IP. Usando cámara local.")
    cap = cv2.VideoCapture(0)

paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        paused = not paused
        print("⏸️ Pausado" if paused else "▶️ Reanudado")
    elif key == 27:  # ESC
        print("🚪 Cerrando por tecla ESC...")
        break

    if paused:
        cv2.putText(frame, "⏸️ PAUSADO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Detección de objetos", frame)
        continue

    results = model(frame, verbose=False)[0]
    boxes = [b for b in results.boxes if int(b.cls[0]) == target_class_id]

    if boxes:
        biggest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, biggest_box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        object_region = frame[y1:y2, x1:x2]
        if object_region.size > 0:
            r, g, b = get_dominant_color(object_region)
            color_name = rgb_to_color_name(r, g, b)
        else:
            color_name = "color desconocido"

        label_text = f"{target_label_es} ({color_name})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Detección de objetos", frame)

# Cierre
cap.release()
cv2.destroyAllWindows()


