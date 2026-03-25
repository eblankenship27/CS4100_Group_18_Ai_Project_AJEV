from ultralytics import YOLO
import cv2
import numpy as np
import mss

model = YOLO("objectTrackingModels/runs/detect/train2/weights/best.pt")

sct = mss.mss()

# monitor = {
#     "top": 100,
#     "left": 100,
#     "width": 800,
#     "height": 600
# }

monitor = sct.monitors[1]

while True:
    screenshot = sct.grab(monitor)

    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    results = model(frame, conf=0.3)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Screen Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()