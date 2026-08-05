"""
Record video from the USB camera for the dataset.

Arducam UC-844 is USB so this uses OpenCV VideoCapture (works on Windows + Pi).

R = start / stop recording (each stop saves a new file)
Q = quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2


# --- settings ---
CLASS_NAME = "football"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / CLASS_NAME / "videos"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIDEO_CODEC = "mp4v"  # works on Windows; use "avc1" if playback looks wrong
WINDOW_NAME = "Hawkeye Record — R record | Q quit"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def next_video_index(output_dir: Path, class_name: str) -> int:
    """Find the next number so we don't overwrite existing videos."""
    existing = list(output_dir.glob(f"{class_name}_*.mp4"))
    if not existing:
        return 1

    indices = []
    for file_path in existing:
        parts = file_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))

    return (max(indices) + 1) if indices else 1


def build_video_path(output_dir: Path, class_name: str, index: int) -> Path:
    return output_dir / f"{class_name}_{index:06d}.mp4"


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open the USB camera. Tries V4L2 first (Pi), then default backend."""
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


def camera_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 120:
        return 30.0
    return float(fps)


def create_writer(path: Path, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Couldn't create video writer for {path}")
    return writer


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def draw_overlay(
    frame,
    recording: bool,
    clip_seconds: float,
    saved_clips: int,
    class_name: str,
):
    display = frame.copy()
    status = "REC" if recording else "READY"
    colour = (0, 0, 255) if recording else (0, 255, 0)

    lines = [
        f"Class: {class_name}",
        f"Status: {status}",
        f"Clip time: {format_duration(clip_seconds)}",
        f"Saved this session: {saved_clips}",
        "R = start/stop | Q = quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
        y += 28

    if recording:
        cv2.circle(display, (display.shape[1] - 30, 30), 10, (0, 0, 255), -1)

    return display


def run_record_loop(cap: cv2.VideoCapture, output_dir: Path, class_name: str) -> int:
    ensure_output_dir(output_dir)
    next_index = next_video_index(output_dir, class_name)
    saved_clips = 0

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Couldn't read from camera.")

    frame_height, frame_width = first_frame.shape[:2]
    fps = camera_fps(cap)

    print(f"Saving to: {output_dir}")
    print(f"Resolution: {frame_width}x{frame_height} @ {fps:.1f} fps")
    print(f"Next file: {class_name}_{next_index:06d}.mp4")
    print("Press R to start/stop recording, Q to quit.")

    recording = False
    writer: cv2.VideoWriter | None = None
    current_path: Path | None = None
    clip_start = 0.0
    clip_frames = 0

    def stop_recording() -> None:
        nonlocal recording, writer, current_path, clip_start, clip_frames, saved_clips, next_index

        if not recording or writer is None:
            return

        writer.release()
        writer = None
        recording = False
        duration = time.monotonic() - clip_start
        saved_clips += 1
        print(
            f"Saved {current_path.name} "
            f"({clip_frames} frames, {format_duration(duration)})"
        )
        next_index += 1
        current_path = None
        clip_frames = 0

    def start_recording() -> None:
        nonlocal recording, writer, current_path, clip_start, clip_frames, next_index

        current_path = build_video_path(output_dir, class_name, next_index)
        writer = create_writer(current_path, frame_width, frame_height, fps)
        recording = True
        clip_start = time.monotonic()
        clip_frames = 0
        print(f"Recording -> {current_path.name}")

    frame = first_frame

    while True:
        if recording and writer is not None:
            writer.write(frame)
            clip_frames += 1

        clip_seconds = time.monotonic() - clip_start if recording else 0.0
        display = draw_overlay(frame, recording, clip_seconds, saved_clips, class_name)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("r"), ord("R")):
            if recording:
                stop_recording()
            else:
                start_recording()
        elif key in (ord("q"), ord("Q")):
            if recording:
                stop_recording()
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            print("Couldn't read from camera.", file=sys.stderr)
            if recording:
                stop_recording()
            break

    if writer is not None:
        writer.release()

    return saved_clips


def release_resources(cap: cv2.VideoCapture) -> None:
    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    cap = None
    try:
        cap = open_camera(CAMERA_INDEX)
        saved = run_record_loop(cap, OUTPUT_DIR, CLASS_NAME)
        print(f"Done. Saved {saved} video(s).")
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            release_resources(cap)


if __name__ == "__main__":
    raise SystemExit(main())
