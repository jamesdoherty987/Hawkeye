"""
Capture stills from the USB camera for the dataset.

Arducam UC-844 is USB so this uses OpenCV VideoCapture (not Picamera2).

SPACE = save frame
Q = quit
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import cv2


# --- settings ---
CLASS_NAME = "football"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / CLASS_NAME / "raw"
CAMERA_INDEX = 0
FRAME_WIDTH = None
FRAME_HEIGHT = None
WINDOW_NAME = "Hawkeye Capture — SPACE save | Q quit"


def ensure_output_dir(path: Path) -> None:
    """Make the output folder if needed."""
    path.mkdir(parents=True, exist_ok=True)


def next_image_index(output_dir: Path, class_name: str) -> int:
    """Find the next number so we don't overwrite existing images."""
    existing = list(output_dir.glob(f"{class_name}_*.jpg"))
    if not existing:
        return 1

    indices = []
    for file_path in existing:
        parts = file_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))

    return (max(indices) + 1) if indices else 1


def build_image_path(output_dir: Path, class_name: str, index: int) -> Path:
    """e.g. football_000001.jpg"""
    return output_dir / f"{class_name}_{index:06d}.jpg"


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
    """Open the USB camera with the right backend for this OS."""
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


def save_frame(frame, output_path: Path) -> bool:
    """Write the frame to disk as a jpg."""
    return bool(cv2.imwrite(str(output_path), frame))


def draw_overlay(frame, captured_count: int, class_name: str):
    """Show count/help text on the preview only (saved images stay clean)."""
    display = frame.copy()
    lines = [
        f"Class: {class_name}",
        f"Captured: {captured_count}",
        "SPACE = save | Q = quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return display


def run_capture_loop(cap: cv2.VideoCapture, output_dir: Path, class_name: str) -> int:
    """Main loop — preview, save on SPACE, quit on Q. Returns how many we saved."""
    ensure_output_dir(output_dir)
    next_index = next_image_index(output_dir, class_name)
    session_saved = 0
    total_on_disk = next_index - 1

    print(f"Saving to: {output_dir}")
    print(f"Starting at: {class_name}_{next_index:06d}.jpg")
    print(f"Already in folder: {total_on_disk}")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Couldn't read from camera.", file=sys.stderr)
            break

        display = draw_overlay(frame, total_on_disk, class_name)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            output_path = build_image_path(output_dir, class_name, next_index)
            if save_frame(frame, output_path):
                print(f"Saved {output_path.name}")
                next_index += 1
                session_saved += 1
                total_on_disk += 1
            else:
                print(f"Failed to save {output_path}", file=sys.stderr)

        elif key in (ord("q"), ord("Q")):
            break

    return session_saved


def release_resources(cap: cv2.VideoCapture) -> None:
    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    cap = None
    try:
        cap = open_camera(CAMERA_INDEX)
        saved = run_capture_loop(cap, OUTPUT_DIR, CLASS_NAME)
        print(f"Done. Saved {saved} new image(s).")
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            release_resources(cap)


if __name__ == "__main__":
    raise SystemExit(main())
