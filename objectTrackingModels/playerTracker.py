from ultralytics import YOLO
import cv2

# this is an example of how the model can be used to find the objects on a screenshot
# when used in the actual game, it must track on a live moving stream
# to test images, add them to the testImages folder and change the frame path

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Read an image (or frame)
frame = cv2.imread("testImages/frame_0186.png")

# Run detection
results = model(frame, conf = 0.1)

# Process detections
detections = []

for r in results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            # which item/class it has found
            "class": int(cls),

            # how confident it is that it is the object
            "confidence": float(conf),

            # the location of the object
            "bbox": [x1, y1, x2, y2]
        })

print("Detections:")
print(detections)