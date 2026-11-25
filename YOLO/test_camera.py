
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery!")

print("Backend:", cap.getBackendName())

success = 0
failed = 0

while True:
    ret, frame = cap.read()

    if ret and frame is not None:
        success += 1
        failed = 0

        cv2.imshow("TEST KAMERY BEZ YOLO", frame)
    else:
        failed += 1
        print(f"Nieudany odczyt: {failed}")

        if failed >= 30:
            break

        time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Poprawnie odczytane klatki:", success)

cap.release()
cv2.destroyAllWindows()
