
from pathlib import Path
import platform
import time

import cv2
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery!")

print("Backend:", cap.getBackendName())
print("Rozdzielczość:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame = None

for attempt in range(30):
    ret, candidate = cap.read()

    if ret and candidate is not None and candidate.size > 0:
        frame = candidate
        print(f"Kamera działa! Próba: {attempt + 1}")
        break

    print(f"Nieudany odczyt {attempt + 1}/30")
    time.sleep(0.1)

if frame is None:
    cap.release()
    raise RuntimeError("Kamera nie zwraca obrazu.")

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH)
print("Urządzenie:", device)
print("Naciśnij Q, aby zakończyć.")

failed_reads = 0

try:
    while True:
        results = model.predict(
            source=frame,
            conf=0.4,
            imgsz=640,
            device=device,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("YOLOv8 - Kamera", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, next_frame = cap.read()

        if ret and next_frame is not None and next_frame.size > 0:
            frame = next_frame
            failed_reads = 0
        else:
            failed_reads += 1
            print(f"Nieudany odczyt: {failed_reads}/30")

            if failed_reads >= 30:
                print("Utracono obraz z kamery.")
                break

            time.sleep(0.1)

finally:
    cap.release()
    cv2.destroyAllWindows()
