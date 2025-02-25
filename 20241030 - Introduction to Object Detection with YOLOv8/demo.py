# This code demonstrates a real-time detection use of the YOLO models

from ultralytics import YOLO

model = YOLO(model="yolov8n.pt")

results = model.predict(source="1", show=True, verbose=True)
