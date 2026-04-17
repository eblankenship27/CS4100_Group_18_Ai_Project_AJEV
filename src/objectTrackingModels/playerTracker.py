from ultralytics import YOLO
import cv2
import numpy as np
import mss

model = YOLO("objectTrackingModels/runs/detect/train2/weights/best.pt")

sct = mss.mss()

monitor = {
    "top": 200,
    "left": 0,
    "width": 600,
    "height": 400
}


while True:
    screenshot = sct.grab(monitor)

    frame = np.array(screenshot)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    results = model(frame, conf=0.1)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Screen Tracking", annotated_frame)

    cv2.moveWindow("YOLO Screen Tracking", 900, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()