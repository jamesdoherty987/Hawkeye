"""
Live football detection with the custom trained model.

Uses models/football_yolov8n.pt (trained on your labeled images).
Falls back to note if the file is missing.

Q to quit.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# --- settings ---
CAMERA_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "football_yolov8n.pt"
IMAGE_SIZE = 320  # smaller = faster on Pi / Mac CPU
CONFIDENCE = 0.35

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Hawkeye Football Detect — Q quit"


def camera_backends() -> list[tuple[str, int]]:
    system = platform.system()
    if system == "Darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("default", cv2.CAP_ANY)]
    if system == "Linux":
        return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
    return [
        ("MSMF", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
        ("default", cv2.CAP_ANY),
    ]


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    for name, backend in camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if FRAME_WIDTH is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if FRAME_HEIGHT is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"Camera {index} opened via {name}.")
            return cap

        cap.release()

    raise RuntimeError(
        f"Couldn't open camera {index}. Check the cable or try another CAMERA_INDEX."
    )


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found: {model_path}\n"
            "Train first with: python training/train_football.py"
        )
    print(f"Loading {model_path}...")
    return YOLO(str(model_path))


def draw_detections(frame, results):
    display = frame.copy()
    count = 0

    if not results or results[0].boxes is None:
        cv2.putText(
            display,
            "football: 0",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display, count

    names = results[0].names or {}

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE:
            continue

        class_id = int(box.cls[0])
        label = names.get(class_id, "football")
        count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        colour = (0, 165, 255)
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

    cv2.putText(
        display,
        f"football: {count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display, count


def run_detection_loop(cap: cv2.VideoCapture, model: YOLO) -> None:
    print("Running custom football model. Press Q to quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Couldn't read from camera.", file=sys.stderr)
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        display, _ = draw_detections(frame, results)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break


def main() -> int:
    cap = None
    try:
        model = load_model(MODEL_PATH)
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
