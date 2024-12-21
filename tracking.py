import cv2
from ultralytics import YOLO
import numpy as np
import sort  # SORT algoritması için

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")  # YOLOv8 model dosyasını belirtin

# SORT tracker oluştur
tracker = sort.Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Güven skoru eşiği
CONFIDENCE_THRESHOLD = 0.5

# Video akışını başlat
cap = cv2.VideoCapture(0)  # Video dosyasını yükleyin veya 0 (kamera) kullanın

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obje tespiti (YOLOv8 ile)
    results = model(frame)

    # Tespit sonuçlarını SORT için uygun formata dönüştür (x1, y1, x2, y2, score)
    detections = []
    for result in results[0].boxes:
        conf = result.conf[0]
        if conf < CONFIDENCE_THRESHOLD:
            continue  # Güven skoru düşükse tespiti atla

        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Tespit koordinatları
        detections.append([x1, y1, x2, y2, float(conf)])

    # SORT algoritması ile tespit edilen objeleri takip et
    if len(detections) > 0:
        tracks = tracker.update(np.array(detections))
    else:
        tracks = np.empty((0, 5))  # Boş bir numpy dizisi döndür

    # Takip edilen objeleri çiz
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Çerçeveyi göster
    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
