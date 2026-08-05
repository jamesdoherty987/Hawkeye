"""
Record video from the USB camera for the dataset.

Arducam UC-844 is USB so this uses OpenCV VideoCapture (works on Windows + Pi).

R = start / stop recording (each stop saves a new file)
Q = quit
"""

from __future__ import annotations

import platform
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
READ_RETRIES = 30  # USB cams on Mac often drop a few frames; don't quit immediately
WARMUP_FRAMES = 10


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


def camera_backends() -> list[tuple[str, int]]:
    """Pick OpenCV backends that match the OS."""
    system = platform.system()
    if system == "Darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("default", cv2.CAP_ANY)]
    if system == "Linux":
        return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
    # Windows
    return [("DirectShow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open the USB camera with the right backend for this OS."""
    last_error = f"Couldn't open camera {index}."

    for name, backend in camera_backends():
        print(f"Trying camera {index} via {name}...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if FRAME_WIDTH is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if FRAME_HEIGHT is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Keep the buffer small so we don't lag behind on USB cams.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ok, frame = read_frame(cap, retries=WARMUP_FRAMES)
        if ok and frame is not None:
            print(f"Camera {index} opened via {name}.")
            return cap

        cap.release()
        last_error = (
            f"Camera {index} opened via {name} but no frames came through. "
            "Close Zoom/FaceTime/browser tabs using the camera, unplug/replug the "
            "Arducam, then try again. If it still fails, change CAMERA_INDEX."
        )

    raise RuntimeError(last_error)


def read_frame(cap: cv2.VideoCapture, retries: int = READ_RETRIES):
    """Read a frame, retrying through short USB / Mac dropouts."""
    for _ in range(max(retries, 1)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.02)
    return False, None


def create_writer(path: Path, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Couldn't create video writer for {path}")
    return writer


def rewrite_with_fps(path: Path, fps: float) -> None:
    """
    Re-save the clip using the real capture rate.

    OpenCV VideoWriter needs an FPS up front, but camera CAP_PROP_FPS is often
    wrong (e.g. 30) while the preview loop only actually grabs ~5–10 fps.
    That mismatch makes playback look sped up.
    """
    temp_path = path.with_name(f"{path.stem}_tmp{path.suffix}")
    reader = cv2.VideoCapture(str(path))
    ok, frame = reader.read()
    if not ok or frame is None:
        reader.release()
        return

    height, width = frame.shape[:2]
    writer = create_writer(temp_path, width, height, fps)
    while ok and frame is not None:
        writer.write(frame)
        ok, frame = reader.read()

    reader.release()
    writer.release()
    temp_path.replace(path)


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

    ok, first_frame = read_frame(cap)
    if not ok or first_frame is None:
        raise RuntimeError(
            "Couldn't read from camera. Close other apps using it, "
            "unplug/replug the cable, then try again."
        )

    frame_height, frame_width = first_frame.shape[:2]
    # Placeholder only — real FPS is measured when the clip stops.
    writer_fps_placeholder = 30.0

    print(f"Saving to: {output_dir}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Next file: {class_name}_{next_index:06d}.mp4")
    print("Press R to start/stop recording, Q to quit.")

    recording = False
    writer: cv2.VideoWriter | None = None
    current_path: Path | None = None
    clip_start = 0.0
    clip_frames = 0

    def stop_recording() -> None:
        nonlocal recording, writer, current_path, clip_start, clip_frames, saved_clips, next_index

        if not recording or writer is None or current_path is None:
            return

        writer.release()
        writer = None
        recording = False
        duration = max(time.monotonic() - clip_start, 1e-6)
        measured_fps = max(clip_frames / duration, 1.0)

        print(f"Fixing playback speed ({measured_fps:.1f} fps)...")
        rewrite_with_fps(current_path, measured_fps)

        saved_clips += 1
        print(
            f"Saved {current_path.name} "
            f"({clip_frames} frames, {format_duration(duration)}, {measured_fps:.1f} fps)"
        )
        next_index += 1
        current_path = None
        clip_frames = 0

    def start_recording() -> None:
        nonlocal recording, writer, current_path, clip_start, clip_frames, next_index

        current_path = build_video_path(output_dir, class_name, next_index)
        writer = create_writer(
            current_path, frame_width, frame_height, writer_fps_placeholder
        )
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

        ok, next_frame = read_frame(cap)
        if not ok or next_frame is None:
            print(
                "Lost camera feed. Close other camera apps, unplug/replug, then retry.",
                file=sys.stderr,
            )
            if recording:
                stop_recording()
            break
        frame = next_frame

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
