"""
Detect + track the football in a video (or live camera).

Uses your trained YOLO model + Ultralytics ByteTrack (built into ultralytics).

For each frame: detect ball → link to same track ID → build path (x, y, time).
Draws trail, speed (px/s), and direction on screen. Saves path as CSV.

Example:
  python capture/track_video.py dataset/football/videos/football_000003.mp4
  python capture/track_video.py dataset/football/videos/football_000003.mp4 --save-video
  python capture/track_video.py --camera
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO


# --- settings ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "football_yolov8n.pt"
EXPORT_VIDEO_DIR = PROJECT_ROOT / "exports" / "football" / "tracked"
EXPORT_PATH_DIR = PROJECT_ROOT / "exports" / "football" / "paths"

IMAGE_SIZE = 640
CONFIDENCE = 0.45  # tracking needs detections often; slightly lower than live-only
TRACKER = "bytetrack.yaml"  # fast default; try "botsort.yaml" if track drops on blur
TRAIL_LENGTH = 60  # max points drawn on trail
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Hawkeye Track — Q quit"


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


def open_camera(index: int = 0) -> cv2.VideoCapture:
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
        f"Couldn't open camera {index}. Check the cable or try another index."
    )


@dataclass
class PathPoint:
    frame: int
    time_s: float
    x: float
    y: float


@dataclass
class BallTrack:
    track_id: int
    points: deque[PathPoint] = field(default_factory=lambda: deque(maxlen=500))

    def add(self, frame: int, time_s: float, x: float, y: float) -> None:
        self.points.append(PathPoint(frame, time_s, x, y))

    def speed_px_per_s(self, min_points: int = 2) -> float:
        """Speed from the last two path points (pixels per second)."""
        if len(self.points) < min_points:
            return 0.0
        a, b = self.points[-2], self.points[-1]
        dt = b.time_s - a.time_s
        if dt <= 0:
            return 0.0
        dist = math.hypot(b.x - a.x, b.y - a.y)
        return dist / dt

    def direction_deg(self, min_points: int = 2) -> float | None:
        """Direction in degrees: 0=right, 90=down (image coords)."""
        if len(self.points) < min_points:
            return None
        a, b = self.points[-2], self.points[-1]
        dx = b.x - a.x
        dy = b.y - a.y
        if dx == 0 and dy == 0:
            return None
        return math.degrees(math.atan2(dy, dx))

    def trail_points(self, max_len: int = TRAIL_LENGTH) -> list[tuple[int, int]]:
        pts = [(int(p.x), int(p.y)) for p in list(self.points)[-max_len:]]
        return pts


def resolve_video_path(raw: Path) -> Path:
    path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if not path.exists():
        alt = PROJECT_ROOT / raw
        if alt.exists():
            return alt.resolve()
        raise FileNotFoundError(f"Video not found: {raw}")
    return path


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first with: python training/train_football.py"
        )
    print(f"Loading {MODEL_PATH}...")
    return YOLO(str(MODEL_PATH))


def pick_ball_box(boxes) -> tuple[int, float, float, float, float, float] | None:
    """
    Choose one ball box from tracker output.
    Returns (track_id, cx, cy, x1, y1, x2, y2) or None.
    """
    best = None
    best_score = -1.0

    for box in boxes:
        if box.id is None:
            continue
        conf = float(box.conf[0])
        if conf < CONFIDENCE:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        # Prefer high confidence; use area as tie-break for small fast ball
        score = conf * 1000 + min(area, 5000) * 0.001
        if score > best_score:
            best_score = score
            best = (int(box.id[0]), cx, cy, x1, y1, x2, y2, conf)

    if best is None:
        return None
    tid, cx, cy, x1, y1, x2, y2, _ = best
    return tid, cx, cy, x1, y1, x2, y2


def draw_track_overlay(
    frame,
    track: BallTrack | None,
    active_track_id: int | None,
    frame_idx: int,
    fps: float,
):
    display = frame.copy()

    if track is not None and len(track.points) >= 1:
        trail = track.trail_points()
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                colour = (0, int(180 + 75 * alpha), int(255 * alpha))
                cv2.line(display, trail[i - 1], trail[i], colour, 2, cv2.LINE_AA)

        last = track.points[-1]
        cx, cy = int(last.x), int(last.y)
        cv2.circle(display, (cx, cy), 6, (0, 255, 255), -1)

        speed = track.speed_px_per_s()
        direction = track.direction_deg()
        dir_text = f"{direction:.0f}°" if direction is not None else "—"

        lines = [
            f"track id: {active_track_id}",
            f"frame: {frame_idx}",
            f"pos: ({cx}, {cy})",
            f"speed: {speed:.0f} px/s",
            f"dir: {dir_text}",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                display,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24
    else:
        cv2.putText(
            display,
            "no ball track",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def save_path_csv(path: Path, track: BallTrack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "x", "y"])
        for p in track.points:
            writer.writerow([p.frame, f"{p.time_s:.4f}", f"{p.x:.2f}", f"{p.y:.2f}"])
    print(f"Path saved: {path} ({len(track.points)} points)")


def next_export_name(prefix: str, suffix: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    existing = list(folder.glob(f"{prefix}_*{suffix}"))
    indices = []
    for p in existing:
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    n = (max(indices) + 1) if indices else 1
    return folder / f"{prefix}_{n:06d}{suffix}"


def run_on_video(
    model: YOLO,
    video_path: Path,
    save_video: bool,
    show: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    out_video_path = None
    if save_video:
        out_video_path = next_export_name(
            video_path.stem + "_tracked", ".mp4", EXPORT_VIDEO_DIR
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    tracks: dict[int, BallTrack] = {}
    active_track_id: int | None = None
    frame_idx = 0
    t0 = time.monotonic()

    print(f"Tracking: {video_path.name} ({width}x{height} @ {fps:.1f} fps)")
    print(f"Tracker: {TRACKER} | conf>={CONFIDENCE} | imgsz={IMAGE_SIZE}")

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1280, width), min(720, height))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        picked = None
        if results and results[0].boxes is not None and len(results[0].boxes):
            picked = pick_ball_box(results[0].boxes)

        if picked is not None:
            tid, cx, cy, x1, y1, x2, y2 = picked
            active_track_id = tid
            if tid not in tracks:
                tracks[tid] = BallTrack(track_id=tid)
            time_s = frame_idx / fps
            tracks[tid].add(frame_idx, time_s, cx, cy)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 165, 255),
                2,
            )

        active = tracks.get(active_track_id) if active_track_id is not None else None
        display = draw_track_overlay(frame, active, active_track_id, frame_idx, fps)

        if writer is not None:
            writer.write(display)
        if show:
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video: {out_video_path}")

    if show:
        cv2.destroyAllWindows()

    # Save path for the track with most points (usually the ball)
    if tracks:
        best_track = max(tracks.values(), key=lambda t: len(t.points))
        csv_path = next_export_name(
            video_path.stem + "_path", ".csv", EXPORT_PATH_DIR
        )
        save_path_csv(csv_path, best_track)
    else:
        print("No ball track found in this video.")

    elapsed = time.monotonic() - t0
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s")
    return frame_idx


def run_on_camera(model: YOLO, camera_index: int) -> int:
    cap = open_camera(camera_index)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracks: dict[int, BallTrack] = {}
    active_track_id: int | None = None
    frame_idx = 0

    print("Live tracking. Press Q to quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        picked = None
        if results and results[0].boxes is not None and len(results[0].boxes):
            picked = pick_ball_box(results[0].boxes)

        if picked is not None:
            tid, cx, cy, x1, y1, x2, y2 = picked
            active_track_id = tid
            if tid not in tracks:
                tracks[tid] = BallTrack(track_id=tid)
            time_s = frame_idx / fps
            tracks[tid].add(frame_idx, time_s, cx, cy)
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 165, 255),
                2,
            )

        active = tracks.get(active_track_id) if active_track_id is not None else None
        display = draw_track_overlay(frame, active, active_track_id, frame_idx, fps)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if tracks:
        best_track = max(tracks.values(), key=lambda t: len(t.points))
        csv_path = next_export_name("live_path", ".csv", EXPORT_PATH_DIR)
        save_path_csv(csv_path, best_track)

    return frame_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track football — trail, speed, direction."
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Path to .mp4 (omit with --camera)",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use live USB camera instead of a video file",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write annotated video to exports/football/tracked/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open preview window (use with --save-video)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = load_model()
        if args.camera:
            run_on_camera(model, args.camera_index)
        else:
            if args.video is None:
                print("Provide a video path or use --camera", file=sys.stderr)
                return 1
            video_path = resolve_video_path(args.video)
            run_on_video(
                model,
                video_path,
                save_video=args.save_video,
                show=not args.no_show,
            )
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
