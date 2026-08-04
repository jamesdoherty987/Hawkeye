import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")

print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    print("Running inference...")
    results = model(frame, imgsz=320, verbose=False)

    print("Inference complete")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
