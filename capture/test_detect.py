"""
Live detection with pretrained YOLOv8n (COCO).

Only showing:
  - person
  - sports ball

No custom training needed — just checking the Pi + camera + YOLO setup works.
Q to quit.
"""

from __future__ import annotations

import sys

import cv2
from ultralytics import YOLO


# --- settings ---
CAMERA_INDEX = 0
MODEL_NAME = "yolov8n.pt"
IMAGE_SIZE = 320  # smaller = faster on Pi
CONFIDENCE = 0.35

# COCO ids: person=0, sports ball=32
TARGET_CLASSES = {
    0: "person",
    32: "sports ball",
}

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "Hawkeye Detect — Q quit"


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open USB camera (V4L2 on the Pi)."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Couldn't open camera {index}. Check the cable or try another CAMERA_INDEX."
        )

    if FRAME_WIDTH is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    if FRAME_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return cap


def load_model(model_name: str = MODEL_NAME) -> YOLO:
    """Load yolov8n — downloads weights the first time."""
    print(f"Loading {model_name}...")
    return YOLO(model_name)


def draw_filtered_detections(frame, results):
    """Draw person / sports ball boxes. Returns annotated frame + counts."""
    display = frame.copy()
    counts = {name: 0 for name in TARGET_CLASSES.values()}

    if not results or results[0].boxes is None:
        return display, counts

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id not in TARGET_CLASSES:
            continue

        conf = float(box.conf[0])
        if conf < CONFIDENCE:
            continue

        label = TARGET_CLASSES[class_id]
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        colour = (0, 255, 0) if class_id == 0 else (0, 165, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            display,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )

    status = " | ".join(f"{name}: {n}" for name, n in counts.items())
    cv2.putText(
        display,
        status,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display, counts


def run_detection_loop(cap: cv2.VideoCapture, model: YOLO) -> None:
    """Read frames and run YOLO until Q is pressed."""
    print("Running. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Couldn't read from camera.", file=sys.stderr)
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=list(TARGET_CLASSES.keys()),
            verbose=False,
        )

        display, _ = draw_filtered_detections(frame, results)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break


def main() -> int:
    cap = None
    try:
        model = load_model(MODEL_NAME)
        cap = open_camera(CAMERA_INDEX)
        run_detection_loop(cap, model)
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
