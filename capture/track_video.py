"""
Detect + track the football in a video (or live camera).

Uses your trained YOLOv8 model + Ultralytics ByteTrack (static sky camera).

Stack (aligned with sports-ball CV best practice):
  - YOLOv8 detect + ByteTrack associate (tracking-by-detection)
  - Sky motion gating (frame diff + background model)
  - Kalman + parabolic flight model for prediction and outlier rejection
  - Offline replay: extend before first detection, fill middle gaps, extend after last

For each frame: motion finds movers → YOLO confirms football → one fused box.

Video: Space = pause | A/D = skip ±1s | Q = quit
When the ball leaves frame you get: Y = instant replay of full path, N = continue
Debug: --show-motion draws faint motion blobs (off by default).

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
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from auto_exposure import SoftwareAutoExposure, read_frame, try_set
from ball_fusion import fuse_ball_detection, fusion_pick_to_tuple
from ball_kalman import BallKalman
from ball_physics import BallFlightModel
from sky_motion import (
    MotionResult,
    SkyMotionDetector,
    draw_motion_overlay,
    nearest_motion_blob,
)


# --- settings ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "football_yolov8n.pt"
EXPORT_VIDEO_DIR = PROJECT_ROOT / "exports" / "football" / "tracked"
EXPORT_PATH_DIR = PROJECT_ROOT / "exports" / "football" / "paths"

IMAGE_SIZE = 640
CONFIDENCE = 0.4  # tracking needs detections often; slightly lower than live-only
TRACKER = str(Path(__file__).resolve().parent / "bytetrack_ball.yaml")
TRAIL_LENGTH = 60  # max points drawn on trail
TRAIL_MAX_SEGMENT_PX = 140  # break trail line on teleports (glitch cleanup)
KALMAN_SEARCH_RADIUS = 120.0  # base px window around predicted ball position
KALMAN_MAX_MISSES = 8  # reset prediction after this many frames without a detection
SEEK_SECONDS = 1.0
REPLAY_SPEED = 1.8  # instant replay playback speed vs video fps
BACKFILL_CONFIDENCE = 0.30  # lower threshold when reconstructing path
BACKFILL_MAX_MISSES = 48  # don't reset mid-segment during backfill
BACKFILL_IMAGE_SIZE = 480  # smaller than live — replay-only, ~40% faster YOLO
BACKFILL_YOLO_BATCH = 4  # batched inference where possible (Pi-friendly)
BACKFILL_EXTEND_S = 4.0  # search this many seconds before/after live detections
MOTION_WARMUP_PAD = 35  # extra frames before search window so motion bg is ready at entry
INTERP_MAX_GAP_FRAMES = 12  # minimum gap-fill span in frames
INTERP_MAX_GAP_S = 0.8  # also allow gaps up to this many seconds
REPLAY_FULL_TRAIL_HOLD_MS = 2500  # pause on last frame showing complete path
MIN_SEGMENT_FRAMES = 3  # minimum tracked frames before offering replay
USE_MOTION = True
SHOW_MOTION = False  # debug: faint motion blobs; fused box is always single
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_INDEX = 1  # match record_video.py / test_detect.py; use --camera-index 0 if needed
WINDOW_NAME = "Hawkeye Track — Space pause | A/D seek | Q quit"


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
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = read_frame(cap, retries=10)
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
    points: deque[PathPoint] = field(default_factory=lambda: deque(maxlen=2000))

    def add(self, frame: int, time_s: float, x: float, y: float) -> bool:
        """Append a point unless this frame is already recorded. Returns True if added."""
        if self.points and self.points[-1].frame == frame:
            return False
        self.points.append(PathPoint(frame, time_s, x, y))
        return True

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

    def trail_points(
        self,
        max_len: int | None = TRAIL_LENGTH,
        max_segment_px: float = TRAIL_MAX_SEGMENT_PX,
    ) -> list[tuple[int, int]]:
        """Trail with gaps broken where a bad detection jumped across the frame."""
        if max_len is None:
            raw = list(self.points)
        else:
            raw = list(self.points)[-max_len:]
        if not raw:
            return []

        segments: list[tuple[int, int]] = [(int(raw[0].x), int(raw[0].y))]
        for point in raw[1:]:
            curr = (int(point.x), int(point.y))
            prev = segments[-1]
            if math.hypot(curr[0] - prev[0], curr[1] - prev[1]) > max_segment_px:
                segments = [curr]
            else:
                segments.append(curr)
        return segments


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
    if not Path(TRACKER).exists():
        raise RuntimeError(f"Tracker config not found: {TRACKER}")
    print(f"Loading {MODEL_PATH}...")
    return YOLO(str(MODEL_PATH))


def reset_yolo_tracker(model: YOLO) -> None:
    """Clear ByteTrack state (e.g. after video seek)."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return
    for tracker in trackers:
        if hasattr(tracker, "reset"):
            tracker.reset()


def reset_session_state(
    kalman_state: KalmanPickState,
    motion_detector: SkyMotionDetector | None,
    model: YOLO,
) -> None:
    kalman_state.reset()
    if motion_detector is not None:
        motion_detector.reset()
    reset_yolo_tracker(model)


@dataclass
class KalmanPickState:
    kalman: BallKalman = field(default_factory=BallKalman)
    flight: BallFlightModel = field(default_factory=BallFlightModel)
    misses: int = 0
    last_reject_reason: str | None = None

    def reset(self) -> None:
        self.kalman.reset()
        self.flight.reset()
        self.misses = 0
        self.last_reject_reason = None

    def begin_frame(self, time_s: float) -> tuple[tuple[float, float] | None, float]:
        if self.flight.can_predict():
            try:
                prediction = self.flight.predict(time_s)
            except (np.linalg.LinAlgError, ValueError):
                prediction = None
            if prediction is not None:
                radius = self.flight.search_radius(base_radius=KALMAN_SEARCH_RADIUS)
                return prediction, radius
        if not self.kalman.initialized:
            return None, KALMAN_SEARCH_RADIUS
        prediction = self.kalman.predict()
        radius = self.kalman.search_radius(base_radius=KALMAN_SEARCH_RADIUS)
        return prediction, radius

    def accept_pick(
        self,
        picked: tuple[int, float, float, float, float, float, float, float] | None,
        time_s: float,
        fps: float,
        relax: bool = False,
        max_misses: int | None = None,
    ) -> tuple[
        tuple[int, float, float, float, float, float, float, float] | None,
        bool,
    ]:
        """
        Gate detections with the flight model, update filters, return
        (accepted_pick_or_none, track_reset_needed).
        """
        miss_limit = max_misses if max_misses is not None else KALMAN_MAX_MISSES

        if picked is None:
            self.misses += 1
            if self.misses > miss_limit:
                self.reset()
                return None, True
            return None, False

        _tid, cx, cy, _x1, _y1, _x2, _y2, _conf = picked
        if relax:
            ok, reason = self.flight.gate_relaxed(time_s, cx, cy, fps)
        else:
            ok, reason = self.flight.gate(time_s, cx, cy, fps)
        if not ok:
            self.last_reject_reason = reason
            self.misses += 1
            if self.misses > miss_limit:
                self.reset()
                return None, True
            return None, False

        self.kalman.update(cx, cy)
        if self.flight.point_count > 0:
            last_t = self.flight._points[-1][0]
            if abs(time_s - last_t) <= 1e-6:
                self.misses = 0
                self.last_reject_reason = None
                return picked, False
        self.flight.add(time_s, cx, cy)
        self.misses = 0
        self.last_reject_reason = None
        return picked, False

    def seed_from_points(
        self,
        points: list[PathPoint],
        fps: float,
        relax: bool = True,
    ) -> None:
        """Warm-start filters from known path samples (bypass gating)."""
        del fps, relax  # kept for call-site compatibility
        self.reset()
        for point in sorted(points, key=lambda p: p.frame):
            self.kalman.update(point.x, point.y)
            if self.flight.point_count == 0:
                self.flight.add(point.time_s, point.x, point.y)
            else:
                last_t = self.flight._points[-1][0]
                if abs(point.time_s - last_t) > 1e-6:
                    self.flight.add(point.time_s, point.x, point.y)
        self.misses = 0
        self.last_reject_reason = None


def run_tracking_iteration(
    frame,
    model: YOLO,
    kalman_state: KalmanPickState,
    motion_detector: SkyMotionDetector | None,
    active_track_id: int | None,
    time_s: float,
    use_motion: bool,
    image_size: int = IMAGE_SIZE,
    enable_crop: bool = True,
    use_track: bool = True,
    min_conf: float | None = None,
) -> tuple[
    MotionResult,
    tuple[float, float] | None,
    float,
    tuple[int, float, float, float, float, float, float, float] | None,
    str | None,
]:
    """Motion + YOLO fusion → single pick per frame."""
    conf = CONFIDENCE if min_conf is None else min_conf
    motion = MotionResult()
    if use_motion and motion_detector is not None:
        motion = motion_detector.process(frame)

    if use_track:
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=conf,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )
    else:
        results = model.predict(
            frame,
            conf=conf,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )

    prediction, search_radius = kalman_state.begin_frame(time_s)
    boxes = results[0].boxes if results and results[0].boxes is not None else None
    prediction_active = (
        kalman_state.flight.can_predict() or kalman_state.kalman.initialized
    )

    fusion = fuse_ball_detection(
        frame,
        model,
        boxes,
        motion,
        prediction,
        search_radius,
        active_track_id,
        prediction_active,
        conf,
        image_size,
        use_motion,
        enable_crop=enable_crop,
        crop_max=2 if enable_crop else 0,
    )

    picked = fusion_pick_to_tuple(fusion) if fusion is not None else None
    source = fusion.source if fusion is not None else None
    return motion, prediction, search_radius, picked, source


def _motion_suggests_detect(
    motion: MotionResult,
    prediction: tuple[float, float] | None,
    search_radius: float,
) -> bool:
    """Skip expensive YOLO on empty sky (motion + prediction gate)."""
    if not motion.ready:
        return False
    if motion.blobs:
        if prediction is None:
            return True
        return (
            nearest_motion_blob(motion.blobs, prediction, search_radius * 1.5)
            is not None
        )
    return prediction is not None


def _fuse_frame_boxes(
    frame,
    model: YOLO,
    kalman_state: KalmanPickState,
    motion: MotionResult,
    boxes,
    active_track_id: int | None,
    time_s: float,
    min_conf: float,
    image_size: int,
    use_motion: bool,
    enable_crop: bool,
) -> tuple[
    tuple[float, float] | None,
    float,
    tuple[int, float, float, float, float, float, float, float] | None,
    str | None,
]:
    prediction, search_radius = kalman_state.begin_frame(time_s)
    prediction_active = (
        kalman_state.flight.can_predict() or kalman_state.kalman.initialized
    )
    fusion = fuse_ball_detection(
        frame,
        model,
        boxes,
        motion,
        prediction,
        search_radius,
        active_track_id,
        prediction_active,
        min_conf,
        image_size,
        use_motion,
        enable_crop=enable_crop,
        crop_max=2 if enable_crop else 0,
    )
    picked = fusion_pick_to_tuple(fusion) if fusion is not None else None
    source = fusion.source if fusion is not None else None
    return prediction, search_radius, picked, source


def _accept_backfill_pick(
    kalman_state: KalmanPickState,
    picked: tuple[int, float, float, float, float, float, float, float] | None,
    time_s: float,
    fps: float,
    *,
    update_flight: bool = True,
) -> tuple[int, float, float] | None:
    if picked is None:
        return None
    if update_flight:
        picked, reset = kalman_state.accept_pick(
            picked,
            time_s,
            fps,
            relax=True,
            max_misses=BACKFILL_MAX_MISSES,
        )
        if reset or picked is None:
            return None
    else:
        _tid, cx, cy, *_rest = picked
        kalman_state.kalman.update(cx, cy)
    tid, cx, cy, *_rest = picked
    return tid, cx, cy


def apply_pick_to_frame(
    frame,
    picked: tuple[int, float, float, float, float, float, float, float] | None,
    fusion_source: str | None,
    show_motion: bool,
    motion: MotionResult,
) -> tuple[int, float, float] | None:
    """Draw one fused box (and optional debug motion blobs)."""
    if show_motion and motion.ready:
        draw_motion_overlay(frame, motion)

    if picked is None:
        return None

    tid, cx, cy, x1, y1, x2, y2, conf = picked
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 165, 255),
        2,
    )
    label = f"ball {conf:.2f}"
    if fusion_source:
        label += f" [{fusion_source}]"
    cv2.putText(
        frame,
        label,
        (int(x1), max(18, int(y1) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    return tid, cx, cy


def draw_prediction_marker(frame, prediction: tuple[float, float] | None, radius: float) -> None:
    if prediction is None:
        return
    px, py = int(prediction[0]), int(prediction[1])
    cv2.circle(frame, (px, py), int(radius), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        (px, py),
        (180, 180, 180),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
        line_type=cv2.LINE_AA,
    )


def seek_video_frame(
    cap: cv2.VideoCapture,
    delta_frames: int,
    total_frames: int,
) -> tuple[bool, object | None, int]:
    """
    Jump forward/back by delta_frames. Returns (ok, frame, 1-based frame index).
    """
    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    target = current + delta_frames
    if total_frames > 0:
        target = max(0, min(target, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None, max(target, 0)
    return True, frame, target + 1


def seek_to_frame(
    cap: cv2.VideoCapture,
    frame_idx: int,
    total_frames: int,
) -> tuple[bool, object | None, int]:
    """Jump to a specific frame index (same indexing as seek_video_frame)."""
    target = max(0, frame_idx - 1)
    if total_frames > 0:
        target = min(target, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None, frame_idx
    return True, frame, target + 1


def merge_track_points(main: BallTrack, backfill: BallTrack) -> int:
    """Insert backfilled points into main track, sorted by frame. Returns count added."""
    by_frame = {p.frame: p for p in main.points}
    added = 0
    for point in backfill.points:
        if point.frame not in by_frame:
            by_frame[point.frame] = point
            added += 1
    ordered = sorted(by_frame.values(), key=lambda p: p.frame)
    maxlen = main.points.maxlen or len(ordered)
    main.points.clear()
    for point in ordered[-maxlen:]:
        main.points.append(point)
    return added


def track_up_to_frame(track: BallTrack, frame_idx: int) -> BallTrack:
    """Temporary track containing only points up to frame_idx (for replay scrub)."""
    partial = BallTrack(track_id=track.track_id)
    for point in track.points:
        if point.frame <= frame_idx:
            partial.add(point.frame, point.time_s, point.x, point.y)
    return partial


def points_in_segment(track: BallTrack, start_frame: int, end_frame: int) -> list[PathPoint]:
    return [p for p in track.points if start_frame <= p.frame <= end_frame]


def track_in_segment(
    track: BallTrack,
    start_frame: int,
    end_frame: int,
) -> BallTrack:
    """Copy only points within a frame range (for segment replay)."""
    clipped = BallTrack(track_id=track.track_id)
    for point in track.points:
        if start_frame <= point.frame <= end_frame:
            clipped.add(point.frame, point.time_s, point.x, point.y)
    return clipped


def compute_backfill_bounds(
    seeds: list[PathPoint],
    total_frames: int,
    fps: float,
) -> tuple[int, int, int, int, int]:
    """
    Live tracking only records first→last detection. Backfill searches wider:

      load_start … search_start … anchor_first … anchor_last … search_end

    load_start includes motion-warmup frames before the search window.
    """
    if not seeds:
        end = max(0, total_frames - 1) if total_frames > 0 else 0
        return 0, 0, end, 0, end

    anchor_first = min(p.frame for p in seeds)
    anchor_last = max(p.frame for p in seeds)
    extend = max(int(round(fps * BACKFILL_EXTEND_S)), 20)
    search_start = max(0, anchor_first - extend)
    load_start = max(0, search_start - MOTION_WARMUP_PAD)
    if total_frames > 0:
        search_end = min(total_frames - 1, anchor_last + extend)
    else:
        search_end = anchor_last + extend
    return load_start, search_start, search_end, anchor_first, anchor_last


def _anchor_pick(point: PathPoint) -> tuple[int, float, float, float, float, float, float, float]:
    return (
        0,
        point.x,
        point.y,
        point.x - 6,
        point.y - 6,
        point.x + 6,
        point.y + 6,
        0.9,
    )


def _interp_max_gap_frames(fps: float) -> int:
    return max(INTERP_MAX_GAP_FRAMES, int(round(fps * INTERP_MAX_GAP_S)))


def _missing_frame_indices(
    point_map: dict[int, PathPoint],
    start_frame: int,
    end_frame: int,
) -> list[int]:
    return [f for f in range(start_frame, end_frame + 1) if f not in point_map]


def _segment_coverage(
    point_map: dict[int, PathPoint],
    start_frame: int,
    end_frame: int,
) -> float:
    total = end_frame - start_frame + 1
    if total <= 0:
        return 1.0
    found = sum(1 for f in point_map if start_frame <= f <= end_frame)
    return found / total


def _load_segment_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    total_frames: int,
) -> dict[int, np.ndarray]:
    """Read segment once into memory (avoids hundreds of random seeks during backfill)."""
    frames: dict[int, np.ndarray] = {}
    ok, frame, fidx = seek_to_frame(cap, start_frame, total_frames)
    if not ok or frame is None:
        return frames

    while fidx <= end_frame:
        frames[fidx] = frame.copy()
        if fidx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return frames


def _interpolate_small_gaps(
    point_map: dict[int, PathPoint],
    start_frame: int,
    end_frame: int,
    max_gap: int = INTERP_MAX_GAP_FRAMES,
) -> int:
    """Linear interpolation between detections for brief misses (Roboflow / tennis trackers)."""
    added = 0
    known = sorted(f for f in point_map if start_frame <= f <= end_frame)
    for i in range(len(known) - 1):
        f0, f1 = known[i], known[i + 1]
        gap = f1 - f0 - 1
        if gap <= 0 or gap > max_gap:
            continue
        p0, p1 = point_map[f0], point_map[f1]
        for fidx in range(f0 + 1, f1):
            if fidx in point_map:
                continue
            t = (fidx - f0) / (f1 - f0)
            point_map[fidx] = PathPoint(
                fidx,
                p0.time_s + (p1.time_s - p0.time_s) * t,
                p0.x + (p1.x - p0.x) * t,
                p0.y + (p1.y - p0.y) * t,
            )
            added += 1
    return added


def _backfill_process_frame(
    frame,
    fidx: int,
    fps: float,
    model: YOLO,
    kalman_state: KalmanPickState,
    motion_detector: SkyMotionDetector | None,
    active_track_id: int | None,
    use_motion: bool,
    image_size: int,
    enable_crop: bool,
    boxes=None,
    motion: MotionResult | None = None,
    *,
    update_flight: bool = True,
) -> tuple[int, float, float] | None:
    time_s = fidx / fps if fps > 0 else fidx / 30.0
    if motion is None:
        motion = MotionResult()
        if use_motion and motion_detector is not None:
            motion = motion_detector.process(frame)

    if boxes is None:
        results = model.predict(
            frame,
            conf=BACKFILL_CONFIDENCE,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )
        boxes = results[0].boxes if results and results[0].boxes is not None else None

    _pred, _radius, picked, _src = _fuse_frame_boxes(
        frame,
        model,
        kalman_state,
        motion,
        boxes,
        active_track_id,
        time_s,
        BACKFILL_CONFIDENCE,
        image_size,
        use_motion,
        enable_crop,
    )
    return _accept_backfill_pick(
        kalman_state, picked, time_s, fps, update_flight=update_flight
    )


def _flush_backfill_batch(
    batch: list[tuple[int, np.ndarray, MotionResult]],
    model: YOLO,
    kalman_state: KalmanPickState,
    fps: float,
    active_track_id: int | None,
    use_motion: bool,
    image_size: int,
    enable_crop: bool,
    point_map: dict[int, PathPoint],
) -> int | None:
    """Run batched YOLO, fuse sequentially (Kalman order matters). Returns last track id."""
    if not batch:
        return active_track_id

    frames = [item[1] for item in batch]
    results = model.predict(
        frames,
        conf=BACKFILL_CONFIDENCE,
        iou=0.5,
        imgsz=image_size,
        verbose=False,
    )

    for i, (fidx, frame, motion) in enumerate(batch):
        time_s = fidx / fps if fps > 0 else fidx / 30.0
        boxes = (
            results[i].boxes
            if i < len(results) and results[i].boxes is not None
            else None
        )
        result = _backfill_process_frame(
            frame,
            fidx,
            fps,
            model,
            kalman_state,
            None,
            active_track_id,
            use_motion,
            image_size,
            enable_crop,
            boxes=boxes,
            motion=motion,
        )
        if result is not None:
            tid, cx, cy = result
            active_track_id = tid
            point_map[fidx] = PathPoint(fidx, time_s, cx, cy)
    return active_track_id


def _nearby_points(
    point_map: dict[int, PathPoint],
    center_frame: int,
    window_frames: int,
) -> list[PathPoint]:
    return sorted(
        (
            p
            for p in point_map.values()
            if abs(p.frame - center_frame) <= window_frames
        ),
        key=lambda p: p.frame,
    )


def _extend_before_first_detection(
    frames: dict[int, np.ndarray],
    model: YOLO,
    search_start: int,
    anchor_first: int,
    fps: float,
    image_size: int,
    enable_crop: bool,
    point_map: dict[int, PathPoint],
) -> None:
    """Second-chance entry search: arc extrapolation backward + YOLO (no motion — time reversed)."""
    if anchor_first <= search_start:
        return

    window = max(int(round(fps * 0.75)), 5)
    early = _nearby_points(point_map, anchor_first, window)
    if len(early) < 2:
        early = sorted(point_map.values(), key=lambda p: p.frame)[: max(2, len(point_map))]
    if not early:
        return

    kalman_state = KalmanPickState()
    kalman_state.seed_from_points(early, fps)

    for fidx in range(anchor_first - 1, search_start - 1, -1):
        if fidx in point_map:
            continue
        frame = frames.get(fidx)
        if frame is None:
            continue
        result = _backfill_process_frame(
            frame,
            fidx,
            fps,
            model,
            kalman_state,
            None,
            0,
            False,
            image_size=image_size,
            enable_crop=enable_crop,
            update_flight=False,
        )
        if result is not None:
            _tid, cx, cy = result
            time_s = fidx / fps if fps > 0 else fidx / 30.0
            point_map[fidx] = PathPoint(fidx, time_s, cx, cy)


def _sync_anchor_to_filters(
    kalman_state: KalmanPickState,
    point: PathPoint,
    fps: float,
) -> None:
    kalman_state.accept_pick(
        _anchor_pick(point),
        point.time_s,
        fps,
        relax=True,
        max_misses=BACKFILL_MAX_MISSES,
    )


def _sweep_forward_motion_yolo(
    frames: dict[int, np.ndarray],
    model: YOLO,
    load_start: int,
    search_start: int,
    search_end: int,
    fps: float,
    use_motion: bool,
    image_size: int,
    enable_crop: bool,
    point_map: dict[int, PathPoint],
    motion_detector: SkyMotionDetector | None,
    kalman_state: KalmanPickState | None = None,
) -> tuple[SkyMotionDetector | None, KalmanPickState]:
    """
    One chronological pass: motion warmup, motion-gated batched YOLO, fusion.
    """
    if kalman_state is None:
        kalman_state = KalmanPickState()
    active_track_id: int | None = None
    batch: list[tuple[int, np.ndarray, MotionResult]] = []

    for fidx in range(load_start, search_end + 1):
        frame = frames.get(fidx)
        if frame is None:
            continue

        # Warm up background model only — no YOLO on pad frames
        if fidx < search_start:
            if use_motion and motion_detector is not None:
                motion_detector.process(frame)
            continue

        if fidx in point_map:
            _sync_anchor_to_filters(kalman_state, point_map[fidx], fps)
            active_track_id = 0
            continue

        motion = MotionResult()
        if use_motion and motion_detector is not None:
            motion = motion_detector.process(frame)

        time_s = fidx / fps if fps > 0 else fidx / 30.0
        prediction, radius = kalman_state.begin_frame(time_s)
        if use_motion and not _motion_suggests_detect(motion, prediction, radius):
            continue

        batch.append((fidx, frame, motion))
        if len(batch) >= BACKFILL_YOLO_BATCH:
            active_track_id = _flush_backfill_batch(
                batch,
                model,
                kalman_state,
                fps,
                active_track_id,
                use_motion,
                image_size,
                enable_crop,
                point_map,
            )
            batch = []

    if batch:
        active_track_id = _flush_backfill_batch(
            batch,
            model,
            kalman_state,
            fps,
            active_track_id,
            use_motion,
            image_size,
            enable_crop,
            point_map,
        )

    return motion_detector, kalman_state


def _gap_fill_with_prediction(
    frames: dict[int, np.ndarray],
    model: YOLO,
    missing_frames: list[int],
    fps: float,
    image_size: int,
    enable_crop: bool,
    point_map: dict[int, PathPoint],
    motion_detector: SkyMotionDetector | None = None,
    use_motion: bool = True,
    kalman_state: KalmanPickState | None = None,
) -> None:
    """YOLO only on frames still missing after the main sweep."""
    if len(point_map) < 2 or not missing_frames:
        return

    if kalman_state is None:
        kalman_state = KalmanPickState()
        kalman_state.seed_from_points(sorted(point_map.values(), key=lambda p: p.frame), fps)

    batch: list[tuple[int, np.ndarray, MotionResult]] = []
    for fidx in missing_frames:
        frame = frames.get(fidx)
        if frame is None:
            continue

        motion = MotionResult()
        if use_motion and motion_detector is not None:
            motion = motion_detector.process(frame)

        time_s = fidx / fps if fps > 0 else fidx / 30.0
        prediction, radius = kalman_state.begin_frame(time_s)
        if use_motion and not _motion_suggests_detect(motion, prediction, radius):
            continue

        batch.append((fidx, frame, motion))
        if len(batch) >= BACKFILL_YOLO_BATCH:
            _flush_backfill_batch(
                batch,
                model,
                kalman_state,
                fps,
                0,
                use_motion,
                image_size,
                enable_crop,
                point_map,
            )
            batch = []

    if batch:
        _flush_backfill_batch(
            batch,
            model,
            kalman_state,
            fps,
            0,
            use_motion,
            image_size,
            enable_crop,
            point_map,
        )


def backfill_ball_path(
    cap: cv2.VideoCapture,
    model: YOLO,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    fps: float,
    use_motion: bool,
    image_size: int,
    enable_crop: bool,
    seed_points: list[PathPoint] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> BallTrack:
    """
    Reconstruct the fullest possible path for a ball segment (optimized).

    1. Load segment once
    2. Single motion-gated forward sweep (batched YOLO at BACKFILL_IMAGE_SIZE)
    3. Backward entry pass only if start still missing
    4. Gap-fill YOLO only on remaining missing frames
    5. Interpolate short gaps
    """
    del image_size  # live imgsz; backfill uses BACKFILL_IMAGE_SIZE
    t0 = time.monotonic()
    bf_imgsz = BACKFILL_IMAGE_SIZE

    def progress(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    seeds = [p for p in (seed_points or []) if start_frame <= p.frame <= end_frame]
    point_map: dict[int, PathPoint] = {p.frame: p for p in seeds}

    load_start, search_start, search_end, anchor_first, anchor_last = (
        compute_backfill_bounds(seeds, total_frames, fps)
    )

    progress(0.05, "loading frames...")
    frames = _load_segment_frames(cap, load_start, search_end, total_frames)
    if not frames:
        track = BallTrack(track_id=-1)
        for point in seeds:
            track.add(point.frame, point.time_s, point.x, point.y)
        return track

    progress(0.15, "detecting path...")
    motion_detector = SkyMotionDetector() if use_motion else None
    motion_detector, kalman_state = _sweep_forward_motion_yolo(
        frames,
        model,
        load_start,
        search_start,
        search_end,
        fps,
        use_motion,
        bf_imgsz,
        enable_crop,
        point_map,
        motion_detector,
    )

    entry_missing = (
        anchor_first > search_start
        and min(point_map.keys(), default=anchor_first) > search_start
    )
    if entry_missing:
        progress(0.55, "finding entry...")
        _extend_before_first_detection(
            frames,
            model,
            search_start,
            anchor_first,
            fps,
            bf_imgsz,
            enable_crop,
            point_map,
        )

    missing = _missing_frame_indices(point_map, search_start, search_end)
    if missing:
        progress(0.7, f"filling {len(missing)} gaps...")
        _gap_fill_with_prediction(
            frames,
            model,
            missing,
            fps,
            bf_imgsz,
            enable_crop,
            point_map,
            motion_detector,
            use_motion,
            kalman_state,
        )

    progress(0.9, "interpolating...")
    _interpolate_small_gaps(
        point_map,
        search_start,
        search_end,
        max_gap=_interp_max_gap_frames(fps),
    )

    track = BallTrack(track_id=-1)
    for fidx in sorted(point_map):
        point = point_map[fidx]
        track.add(point.frame, point.time_s, point.x, point.y)
    if track.points:
        track.track_id = 0

    elapsed = time.monotonic() - t0
    span = search_end - search_start + 1
    cov = _segment_coverage(point_map, search_start, search_end)
    print(
        f"Backfill done in {elapsed:.1f}s — {len(track.points)} pts, "
        f"{cov * 100:.0f}% coverage over {span} frames (imgsz={bf_imgsz})"
    )
    return track


def draw_banner(display, text: str, y: int = 52) -> None:
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    x2 = min(display.shape[1] - 8, 14 + text_size[0] + 10)
    cv2.rectangle(display, (8, y - 24), (x2, y + 10), (0, 0, 0), -1)
    cv2.putText(
        display,
        text,
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def show_status_frame(message: str, width: int, height: int) -> None:
    """Brief full-screen status (e.g. while backfilling path)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    draw_banner(frame, message, y=height // 2)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def prompt_replay_choice(display) -> bool | None:
    """Block until Y/N/Q. Returns True=replay, False=skip, None=quit."""
    while True:
        overlay = display.copy()
        draw_banner(overlay, "Ball left frame — Y = instant replay | N = continue | Q = quit")
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N")):
            return False
        if key in (ord("q"), ord("Q")):
            return None


def play_instant_replay(
    cap: cv2.VideoCapture,
    track: BallTrack,
    start_frame: int,
    end_frame: int,
    fps: float,
    total_frames: int,
) -> bool:
    """Scrub segment with growing trail, then hold full path on last frame."""
    delay_ms = max(1, int(1000.0 / max(fps, 1.0) / REPLAY_SPEED))
    ok, frame, fidx = seek_to_frame(cap, start_frame, total_frames)
    if not ok or frame is None:
        return False

    quit_requested = False
    while fidx <= end_frame:
        partial = track_up_to_frame(track, fidx)
        display = draw_track_overlay(
            frame,
            partial,
            track.track_id,
            fidx,
            fps,
            fusion_source="replay",
            full_trail=True,
        )
        draw_banner(display, "INSTANT REPLAY")
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), ord("Q")):
            quit_requested = True
            break
        if fidx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if not quit_requested and track.points:
        ok, hold_frame, hold_idx = seek_to_frame(cap, end_frame, total_frames)
        if ok and hold_frame is not None:
            hold_until = time.monotonic() + REPLAY_FULL_TRAIL_HOLD_MS / 1000.0
            while time.monotonic() < hold_until:
                display = draw_track_overlay(
                    hold_frame,
                    track,
                    track.track_id,
                    hold_idx,
                    fps,
                    fusion_source="replay",
                    full_trail=True,
                )
                draw_banner(
                    display,
                    f"FULL PATH — {len(track.points)} pts | any key to continue",
                )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), ord("Q")):
                    quit_requested = True
                    break
                if key != 255:
                    break

    for _ in range(3):
        cv2.waitKey(1)
    return quit_requested


def run_segment_replay(
    cap: cv2.VideoCapture,
    model: YOLO,
    track: BallTrack,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    use_motion: bool,
    image_size: int,
    enable_crop: bool,
) -> bool:
    """Backfill full path, play replay. Returns True if user quit."""
    def on_progress(fraction: float, message: str) -> None:
        show_status_frame(
            f"Building path… {int(fraction * 100)}% — {message}",
            width,
            height,
        )

    seeds = points_in_segment(track, start_frame, end_frame)
    load_start, search_start, search_end, anchor_first, anchor_last = (
        compute_backfill_bounds(seeds, total_frames, fps)
    )
    extend_before = anchor_first - search_start
    extend_after = search_end - anchor_last
    print(
        f"Backfill anchors {anchor_first}–{anchor_last} "
        f"(live {start_frame}–{end_frame}), "
        f"search {search_start}–{search_end} "
        f"(motion from f{load_start}, +{extend_before}f before, +{extend_after}f after)"
    )
    backfill = backfill_ball_path(
        cap,
        model,
        start_frame,
        end_frame,
        total_frames,
        fps,
        use_motion,
        image_size,
        enable_crop,
        seed_points=seeds,
        on_progress=on_progress,
    )
    reset_yolo_tracker(model)
    added = merge_track_points(track, backfill)

    # Replay only this segment — not the full session track (avoids playing whole video
    # after A/D seeks left points from earlier in the file).
    replay_track = backfill if backfill.points else track_in_segment(
        track, search_start, search_end
    )
    replay_start = search_start
    replay_end = search_end
    span = replay_end - replay_start + 1
    print(
        f"Replay: {len(replay_track.points)} pts in segment, "
        f"playing frames {replay_start}–{replay_end} ({span}f, {added} new from backfill)"
    )
    return play_instant_replay(
        cap, replay_track, replay_start, replay_end, fps, total_frames
    )


def draw_playback_hud(
    display,
    *,
    video_fps: float,
    proc_fps: float,
    frame_idx: int,
    total_frames: int,
    show_seek_hints: bool,
    paused: bool = False,
) -> None:
    """On-screen help + speed indicator (processing is often slower than video fps)."""
    h = display.shape[0]
    if show_seek_hints:
        hint = (
            "PAUSED — Space resume | A/D skip 1s | Q quit"
            if paused
            else "Space pause | A/D skip 1s | Q quit"
        )
        cv2.putText(
            display,
            hint,
            (10, h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    speed_note = ""
    if video_fps > 0 and proc_fps > 0 and proc_fps < video_fps * 0.85:
        speed_note = " (processing lag — use --no-show --save-video)"

    cv2.putText(
        display,
        f"proc {proc_fps:.1f} fps | video {video_fps:.0f} fps{speed_note}",
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if total_frames > 0:
        cv2.putText(
            display,
            f"frame {frame_idx}/{total_frames}",
            (display.shape[1] - 200, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def draw_track_overlay(
    frame,
    track: BallTrack | None,
    active_track_id: int | None,
    frame_idx: int,
    fps: float,
    fusion_source: str | None = None,
    reject_reason: str | None = None,
    *,
    full_trail: bool = False,
):
    display = frame.copy()

    if track is not None and len(track.points) >= 1:
        trail = track.trail_points(max_len=None if full_trail else TRAIL_LENGTH)
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
        if fusion_source:
            lines.append(f"fuse: {fusion_source}")
        if reject_reason:
            lines.append(f"rejected: {reject_reason}")
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
    use_motion: bool = USE_MOTION,
    show_motion: bool = SHOW_MOTION,
    image_size: int = IMAGE_SIZE,
    enable_crop: bool = True,
    enable_replay: bool = True,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    seek_frames = max(1, int(round(fps * SEEK_SECONDS)))

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
    kalman_state = KalmanPickState()
    motion_detector = SkyMotionDetector() if use_motion else None

    motion_label = "on" if use_motion else "off"
    print(f"Tracking: {video_path.name} ({width}x{height} @ {fps:.1f} fps)")
    print(
        f"Tracker: {Path(TRACKER).name} | conf>={CONFIDENCE} | imgsz={image_size} | "
        f"Kalman + flight arc fusion {motion_label}"
    )
    replay_label = "on loss" if enable_replay and show else "off"
    print(f"Space = pause | A/D = skip ±1s | replay prompt {replay_label}")

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1280, width), min(720, height))

    frame = None
    fusion_source: str | None = None
    paused = False
    last_display = None
    proc_times: deque[float] = deque(maxlen=30)

    # Ball in-frame segment (entry → exit) for optional replay
    segment_start: int | None = None
    segment_end: int | None = None
    segment_tid: int | None = None

    while True:
        if frame is None and not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not paused:
            frame_start = time.monotonic()
            time_s = frame_idx / fps if fps > 0 else frame_idx / 30.0
            motion, prediction, search_radius, picked, fusion_source = (
                run_tracking_iteration(
                    frame,
                    model,
                    kalman_state,
                    motion_detector,
                    active_track_id,
                    time_s,
                    use_motion,
                    image_size=image_size,
                    enable_crop=enable_crop,
                )
            )
            proc_times.append(time.monotonic() - frame_start)
            proc_fps = (
                len(proc_times) / sum(proc_times)
                if proc_times and sum(proc_times) > 0
                else 0.0
            )

            picked, reset_track = kalman_state.accept_pick(picked, time_s, fps)
            draw_prediction_marker(frame, prediction, search_radius)

            update = apply_pick_to_frame(
                frame, picked, fusion_source, show_motion, motion
            )
            if update is not None:
                tid, cx, cy = update
                if tid not in tracks:
                    tracks[tid] = BallTrack(track_id=tid)
                if segment_start is None:
                    segment_start = frame_idx
                segment_end = frame_idx
                segment_tid = tid
                active_track_id = tid
                tracks[tid].add(frame_idx, time_s, cx, cy)
            elif reset_track:
                active_track_id = None

            active = tracks.get(active_track_id) if active_track_id is not None else None
            display = draw_track_overlay(
                frame,
                active,
                active_track_id,
                frame_idx,
                fps,
                fusion_source,
                reject_reason=kalman_state.last_reject_reason,
            )
            draw_playback_hud(
                display,
                video_fps=fps,
                proc_fps=proc_fps,
                frame_idx=frame_idx,
                total_frames=total_frames,
                show_seek_hints=show,
                paused=False,
            )
            last_display = display

            if writer is not None:
                writer.write(display)

            if reset_track and segment_start is not None and segment_end is not None:
                resume_frame = frame_idx
                seg_start = segment_start
                seg_end = segment_end
                seg_tid = segment_tid
                segment_start = None
                segment_end = None
                segment_tid = None

                long_enough = (
                    seg_tid is not None
                    and seg_end - seg_start >= MIN_SEGMENT_FRAMES - 1
                )
                if enable_replay and show and long_enough:
                    print(
                        f"Ball left frame (tracked {seg_start}–{seg_end}). "
                        "Instant replay?"
                    )
                    choice = prompt_replay_choice(last_display)
                    if choice is None:
                        break
                    if choice:
                        if run_segment_replay(
                            cap,
                            model,
                            tracks[seg_tid],
                            seg_start,
                            seg_end,
                            total_frames,
                            fps,
                            width,
                            height,
                            use_motion,
                            image_size,
                            enable_crop,
                        ):
                            break
                    ok, frame, frame_idx = seek_to_frame(
                        cap, resume_frame, total_frames
                    )
                    if not ok or frame is None:
                        break
                    reset_session_state(kalman_state, motion_detector, model)
        elif last_display is not None:
            display = last_display.copy()
            draw_playback_hud(
                display,
                video_fps=fps,
                proc_fps=0.0,
                frame_idx=frame_idx,
                total_frames=total_frames,
                show_seek_hints=show,
                paused=True,
            )
        else:
            display = None

        if show and display is not None:
            cv2.imshow(WINDOW_NAME, display)
            wait_ms = 30 if paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("a"), ord("A"), ord("d"), ord("D")):
                delta = -seek_frames if key in (ord("a"), ord("A")) else seek_frames
                ok, seek_frame, new_idx = seek_video_frame(cap, delta, total_frames)
                if ok and seek_frame is not None:
                    frame = seek_frame
                    frame_idx = new_idx
                    active_track_id = None
                    segment_start = None
                    segment_end = None
                    segment_tid = None
                    reset_session_state(kalman_state, motion_detector, model)
                    paused = False
                    direction = "back" if delta < 0 else "forward"
                    print(f"Jumped {direction} to frame {frame_idx}/{total_frames}")

        if not paused:
            frame = None

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
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")
    if fps > 0 and avg_fps < fps * 0.85:
        print(
            f"Note: processing ({avg_fps:.1f} fps) is slower than video ({fps:.0f} fps) "
            "— preview looks like slow motion. Use --no-show --save-video."
        )
    return frame_idx


def run_on_camera(
    model: YOLO,
    camera_index: int,
    use_motion: bool = USE_MOTION,
    show_motion: bool = SHOW_MOTION,
    image_size: int = IMAGE_SIZE,
    enable_crop: bool = True,
) -> int:
    cap = open_camera(camera_index)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    auto_exp = SoftwareAutoExposure(cap)
    auto_exp.settle()

    tracks: dict[int, BallTrack] = {}
    active_track_id: int | None = None
    frame_idx = 0
    kalman_state = KalmanPickState()
    motion_detector = SkyMotionDetector() if use_motion else None

    motion_label = "on" if use_motion else "off"
    print(f"Live tracking (sky auto-exposure, motion {motion_label}). Space = pause | Q = quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False
    last_display = None

    while True:
        if not paused:
            ok, frame = read_frame(cap)
            if not ok or frame is None:
                break

            frame = auto_exp.process(frame)
            time_s = frame_idx / fps if fps > 0 else frame_idx / 30.0
            frame_idx += 1
            motion, prediction, search_radius, picked, fusion_source = (
                run_tracking_iteration(
                    frame,
                    model,
                    kalman_state,
                    motion_detector,
                    active_track_id,
                    time_s,
                    use_motion,
                    image_size=image_size,
                    enable_crop=enable_crop,
                )
            )

            picked, reset_track = kalman_state.accept_pick(picked, time_s, fps)
            if reset_track:
                active_track_id = None
            draw_prediction_marker(frame, prediction, search_radius)

            update = apply_pick_to_frame(
                frame, picked, fusion_source, show_motion, motion
            )
            if update is not None:
                tid, cx, cy = update
                active_track_id = tid
                if tid not in tracks:
                    tracks[tid] = BallTrack(track_id=tid)
                tracks[tid].add(frame_idx, time_s, cx, cy)

            active = tracks.get(active_track_id) if active_track_id is not None else None
            last_display = draw_track_overlay(
                frame,
                active,
                active_track_id,
                frame_idx,
                fps,
                fusion_source,
                reject_reason=kalman_state.last_reject_reason,
            )
            cv2.putText(
                last_display,
                auto_exp.status_text(),
                (10, last_display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            frame_idx += 1

        if last_display is not None:
            display = last_display.copy()
            if paused:
                draw_playback_hud(
                    display,
                    video_fps=fps,
                    proc_fps=0.0,
                    frame_idx=frame_idx,
                    total_frames=0,
                    show_seek_hints=True,
                    paused=True,
                )
            cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" "):
            paused = not paused

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
        default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
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
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Disable sky motion assist (YOLO + Kalman only)",
    )
    parser.add_argument(
        "--show-motion",
        action="store_true",
        help="Show green motion blob debug overlays",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Disable Y/N replay prompt when ball leaves frame",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_motion = not args.no_motion
    show_motion = args.show_motion
    try:
        model = load_model()
        if args.camera:
            run_on_camera(
                model,
                args.camera_index,
                use_motion=use_motion,
                show_motion=show_motion,
            )
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
                use_motion=use_motion,
                show_motion=show_motion,
                enable_replay=not args.no_replay,
            )
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
