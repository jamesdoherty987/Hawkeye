"""
stereo_record.py — Dual-camera ball tracking, path building, and automated clip saving.

State machine
─────────────
  IDLE  (green border)
      Both cameras live. Detects ball using motion → YOLO → Kalman.
      Keeps a rolling pre-roll ring buffer so the ball's entry is never cut off.

  TRACKING  (white border + "● REC")
      Ball detected. Recording every frame to memory.
      Continues until the ball has been lost from BOTH cameras for > LOST_S seconds.

  PROCESSING  (red overlay)
      Ball gone. Re-runs YOLO on every buffered frame to rebuild the complete
      flight path, fills small gaps via linear interpolation, renders the
      full path onto the clip, then saves a side-by-side annotated MP4.
      Returns to IDLE when done.

Saved to: exports/stereo/recordings/stereo_YYYYMMDD_HHMMSS.mp4

Usage
─────
  python capture/stereo_record.py --list
  python capture/stereo_record.py --left 0 --right 1
  python capture/stereo_record.py --left 0 --right 1 \\
      --model models/football_yolov8n.pt --baseline 6.5
  python capture/stereo_record.py --left 0 --right 1 --one-camera  # trigger on any camera

Keys: Q = quit
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections import deque
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from auto_exposure import try_set
from ball_kalman import BallKalman
from sky_motion import MotionResult, SkyMotionDetector, box_overlaps_motion
import stereo_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "exports" / "stereo" / "recordings"
COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32

# ── Detection ────────────────────────────────────────────────────────────────
IMAGE_SIZE = 640
CONFIDENCE = 0.35
BACKFILL_CONF = 0.25         # lower confidence for the post-recording re-pass
HIGH_CONF_NO_MOTION = 0.72   # accept YOLO det even without motion at this conf
MOTION_GATE_OVERLAP = 0.10   # minimum IoU between det and nearest motion blob

# ── Recording ────────────────────────────────────────────────────────────────
PRE_ROLL_S = 1.5             # pre-roll ring-buffer duration (seconds)
LOST_S = 0.8                 # seconds both cameras must miss ball before processing
MIN_TRACK_FRAMES = 8         # discard clips shorter than this many frames
MAX_BUFFER_S = 20.0          # hard cap on in-memory recording

# ── Path building ─────────────────────────────────────────────────────────────
INTERP_MAX_GAP = 8           # max frame gap to fill by linear interpolation

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW = "Hawkeye Stereo Recorder — Q quit"
TRAIL_L = (0, 255, 255)      # left camera trail: cyan
TRAIL_R = (255, 80, 200)     # right camera trail: magenta
IDLE_COL = (0, 200, 50)      # green border
TRACK_COL = (255, 255, 255)  # white border
PROC_COL = (0, 0, 180)       # red tint
BORDER_PX = 10
OVERLAY_A = 0.40             # red overlay transparency


class State(Enum):
    IDLE = auto()
    TRACKING = auto()
    PROCESSING = auto()


# ─── Camera helpers ──────────────────────────────────────────────────────────

def _backends() -> list[tuple[str, int]]:
    s = platform.system()
    if s == "Darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("default", cv2.CAP_ANY)]
    if s == "Linux":
        return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
    return [("MSMF", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]


def open_camera(index: int) -> cv2.VideoCapture:
    last_err = f"Could not open camera {index}."
    for name, backend in _backends():
        print(f"  Trying camera {index} via {name}...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = False, None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None and frame.any():
                break
            time.sleep(0.05)
        if ok and frame is not None and frame.any():
            h, w = frame.shape[:2]
            print(f"  Camera {index} OK via {name} — {w}×{h}")
            return cap
        cap.release()
        last_err = f"Camera {index} via {name}: no valid frames. Close other apps and retry."
    raise RuntimeError(last_err)


def read_cam(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    return (True, frame) if (ok and frame is not None) else (False, None)


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model(model_str: str) -> tuple[YOLO, list[int] | None]:
    is_coco = (Path(model_str).name == COCO_MODEL)
    print(f"Loading {'COCO' if is_coco else 'custom'} model: {model_str}")
    model = YOLO(model_str)
    classes: list[int] | None = [SPORTS_BALL_CLASS] if is_coco else None
    print(f"  Class filter: {classes}")
    return model, classes


def _parse_best(result) -> tuple[float, float, float, float, float, float, float] | None:
    """Return (cx, cy, x1, y1, x2, y2, conf) for the top-confidence box, or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best: tuple | None = None
    best_c = -1.0
    for box in result.boxes:
        c = float(box.conf[0])
        if c <= best_c:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_c = c
        best = ((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2, c)
    return best


def _motion_gate(
    raw: tuple | None,
    blobs: list,
) -> tuple | None:
    """Drop a detection that doesn't overlap motion unless its confidence is very high."""
    if raw is None:
        return None
    cx, cy, x1, y1, x2, y2, c = raw
    if not blobs:
        return raw if c >= HIGH_CONF_NO_MOTION else None
    if box_overlaps_motion(x1, y1, x2, y2, blobs) >= MOTION_GATE_OVERLAP:
        return raw
    return raw if c >= HIGH_CONF_NO_MOTION else None


# ─── Live detection (batched — both cameras in one YOLO call) ────────────────

def detect_pair(
    model: YOLO,
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    mr_l: MotionResult,
    mr_r: MotionResult,
    kalman_l: BallKalman,
    kalman_r: BallKalman,
    conf: float,
    classes: list[int] | None,
) -> tuple[tuple | None, tuple | None]:
    """
    Batch-infer both frames in a single model.predict() call.
    Motion gates the result: detections not near any motion blob are dropped
    unless they exceed HIGH_CONF_NO_MOTION.
    Returns (det_left, det_right) — each is (cx, cy, x1, y1, x2, y2, conf) or None.
    """
    blobs_l = mr_l.blobs if mr_l.ready else []
    blobs_r = mr_r.blobs if mr_r.ready else []
    need_l = bool(blobs_l) or kalman_l.initialized
    need_r = bool(blobs_r) or kalman_r.initialized
    if not (need_l or need_r):
        return None, None

    kw: dict = dict(conf=conf, imgsz=IMAGE_SIZE, verbose=False)
    if classes:
        kw["classes"] = classes
    results = model.predict([frame_l, frame_r], **kw)
    det_l = _motion_gate(_parse_best(results[0]), blobs_l) if need_l else None
    det_r = _motion_gate(_parse_best(results[1]), blobs_r) if need_r else None
    return det_l, det_r


# ─── Backfill: re-run YOLO on every recorded frame to build complete paths ───

def build_paths(
    frames_l: list[np.ndarray],
    frames_r: list[np.ndarray],
    model: YOLO,
    classes: list[int] | None,
    on_progress=None,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """
    Re-infer YOLO on every frame pair (batched) using a lower confidence threshold.
    Motion gating is applied using a fresh SkyMotionDetector to preserve the background model.
    Returns dicts mapping frame_idx → (cx, cy) for left and right cameras.
    """
    motion_l = SkyMotionDetector()
    motion_r = SkyMotionDetector()
    dets_l: dict[int, tuple[float, float]] = {}
    dets_r: dict[int, tuple[float, float]] = {}
    n = min(len(frames_l), len(frames_r))

    kw: dict = dict(conf=BACKFILL_CONF, imgsz=IMAGE_SIZE, verbose=False)
    if classes:
        kw["classes"] = classes

    for i in range(n):
        fl, fr = frames_l[i], frames_r[i]
        mr_l = motion_l.process(fl)
        mr_r = motion_r.process(fr)

        results = model.predict([fl, fr], **kw)
        raw_l = _parse_best(results[0])
        raw_r = _parse_best(results[1])

        det_l = _motion_gate(raw_l, mr_l.blobs if mr_l.ready else [])
        det_r = _motion_gate(raw_r, mr_r.blobs if mr_r.ready else [])
        if det_l:
            dets_l[i] = (det_l[0], det_l[1])
        if det_r:
            dets_r[i] = (det_r[0], det_r[1])

        if on_progress and i % 5 == 0:
            on_progress(i, n)

    if on_progress:
        on_progress(n, n)
    return dets_l, dets_r


def interpolate_path(
    dets: dict[int, tuple[float, float]],
    n_frames: int,
    max_gap: int = INTERP_MAX_GAP,
) -> dict[int, tuple[float, float]]:
    """Fill small detection gaps with linear interpolation."""
    if not dets:
        return {}
    result = dict(dets)
    keys = sorted(dets)
    for a, b in zip(keys[:-1], keys[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            x0, y0 = dets[a]
            x1, y1 = dets[b]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                result[a + k] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return result


# ─── Display helpers ─────────────────────────────────────────────────────────

def draw_trail(
    frame: np.ndarray,
    path: dict[int, tuple[float, float]],
    color: tuple[int, int, int],
    up_to: int | None = None,
    current_idx: int | None = None,
) -> None:
    """
    Draw the ball path as a line + dots.
    up_to:       if set, only draw points with idx <= up_to (growing trail on live display).
                 None = draw ALL points (full path on saved video).
    current_idx: if set, highlight that frame's ball position with a bigger ring.
    """
    pts = [
        (i, int(cx), int(cy))
        for i, (cx, cy) in sorted(path.items())
        if up_to is None or i <= up_to
    ]
    if len(pts) >= 2:
        for (_, ax, ay), (_, bx, by) in zip(pts[:-1], pts[1:]):
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i, x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
    if current_idx is not None:
        for i, x, y in pts:
            if i == current_idx:
                cv2.circle(frame, (x, y), 9, color, 3, cv2.LINE_AA)
                break


def paint_det(
    frame: np.ndarray,
    det: tuple | None,
    color: tuple[int, int, int],
) -> None:
    if det is None:
        return
    cx, cy, x1, y1, x2, y2, conf = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"{conf:.2f}", (int(x1), max(int(y1) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def make_panel(frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
    """Stack both camera frames side-by-side at a common height."""
    h = min(frame_l.shape[0], frame_r.shape[0])
    def fit(f: np.ndarray) -> np.ndarray:
        if f.shape[0] == h:
            return f
        s = h / f.shape[0]
        return cv2.resize(f, (int(f.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    return np.hstack([fit(frame_l), fit(frame_r)])


def add_state_overlay(combo: np.ndarray, state: State, label: str = "") -> None:
    """Apply coloured border (IDLE/TRACKING) or red tint (PROCESSING) to the panel."""
    h, w = combo.shape[:2]
    if state == State.IDLE:
        cv2.rectangle(combo, (0, 0), (w - 1, h - 1), IDLE_COL, BORDER_PX)
        cv2.putText(combo, label or "LOOKING FOR BALL", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, IDLE_COL, 2, cv2.LINE_AA)
    elif state == State.TRACKING:
        cv2.rectangle(combo, (0, 0), (w - 1, h - 1), TRACK_COL, BORDER_PX)
        cv2.putText(combo, label or "● RECORDING", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 255), 2, cv2.LINE_AA)
    elif state == State.PROCESSING:
        overlay = combo.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), PROC_COL, -1)
        cv2.addWeighted(overlay, OVERLAY_A, combo, 1 - OVERLAY_A, 0, combo)
        txt = label or "PROCESSING…"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(combo, txt, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


# ─── Save annotated clip ─────────────────────────────────────────────────────

def save_clip(
    frames_l: list[np.ndarray],
    frames_r: list[np.ndarray],
    path_l: dict[int, tuple[float, float]],
    path_r: dict[int, tuple[float, float]],
    fps: float,
) -> Path:
    """
    Render and save a side-by-side annotated MP4.
    The FULL ball path (both cameras) is drawn on EVERY frame of the saved video —
    the entire flight arc is visible from the very first frame.
    The ball's position at that frame is highlighted with a larger ring.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"stereo_{ts}.mp4"
    n = min(len(frames_l), len(frames_r))
    if n == 0:
        raise ValueError("Empty frame buffer — nothing to save")

    # Output dimensions: both cameras resized to the same height, placed side-by-side
    h = min(frames_l[0].shape[0], frames_r[0].shape[0])
    w_l = int(frames_l[0].shape[1] * h / frames_l[0].shape[0])
    w_r = int(frames_r[0].shape[1] * h / frames_r[0].shape[0])

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_l + w_r, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out}")

    for i in range(n):
        fl = cv2.resize(frames_l[i], (w_l, h))
        fr = cv2.resize(frames_r[i], (w_r, h))

        # Full path (all points) on every frame; current position highlighted
        draw_trail(fl, path_l, TRAIL_L, up_to=None, current_idx=i)
        draw_trail(fr, path_r, TRAIL_R, up_to=None, current_idx=i)

        cv2.putText(fl, f"LEFT  f{i}/{n}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRAIL_L, 2, cv2.LINE_AA)
        cv2.putText(fr, f"RIGHT f{i}/{n}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRAIL_R, 2, cv2.LINE_AA)
        writer.write(np.hstack([fl, fr]))

    writer.release()
    print(f"  ✓ Saved → {out}")
    return out


# ─── Main loop ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-camera stereo ball recorder")
    parser.add_argument("--left",  type=int, default=0, help="Left camera index")
    parser.add_argument("--right", type=int, default=1, help="Right camera index")
    parser.add_argument(
        "--model", type=str, default=COCO_MODEL,
        help=f"Model file. Default {COCO_MODEL} (COCO, any environment). "
             "Use models/football_yolov8n.pt for outdoor/sky tracking.",
    )
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Detection confidence (default {CONFIDENCE})")
    parser.add_argument("--baseline", type=float, default=cfg.BASELINE_M,
                        help="Camera separation in metres (informational only, default 1.0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Camera FPS hint for buffer sizing (default 30)")
    parser.add_argument("--one-camera", action="store_true",
                        help="Start recording when ANY camera sees the ball "
                             "(default: require both cameras to see it)")
    parser.add_argument("--list", action="store_true",
                        help="List available camera indices 0–5 and exit")
    args = parser.parse_args()

    if args.list:
        print("Probing cameras 0–5:")
        for i in range(6):
            try:
                cap = open_camera(i)
                print(f"  → {i} OK")
                cap.release()
                time.sleep(0.3)
            except RuntimeError as exc:
                print(f"  → {i} FAIL: {exc}")
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1
    if not 0.0 < args.conf <= 1.0:
        print("ERROR: --conf must be in (0, 1]", file=sys.stderr)
        return 1

    model_path = args.model
    if Path(model_path).name != COCO_MODEL:
        full = (PROJECT_ROOT / model_path) if not Path(model_path).is_absolute() else Path(model_path)
        if not full.exists():
            print(f"ERROR: model not found: {full}", file=sys.stderr)
            return 1
        model_path = str(full)

    model, classes = load_model(model_path)
    require_both = not args.one_camera

    print(
        f"\nBaseline={args.baseline:.2f} m  |  left={args.left}  right={args.right}\n"
        f"Trigger: {'BOTH cameras' if require_both else 'ANY camera'}  |  conf={args.conf:.2f}\n"
        f"Recordings → {OUT_DIR}\n"
        "Q = quit\n"
    )

    cap_l = cap_r = None
    try:
        print(f"Opening left camera ({args.left})...")
        cap_l = open_camera(args.left)
        time.sleep(0.5)
        print(f"Opening right camera ({args.right})...")
        cap_r = open_camera(args.right)

        # Motion detectors: run continuously — never reset between clips so the
        # background model stays warm and detections start instantly.
        motion_l = SkyMotionDetector()
        motion_r = SkyMotionDetector()

        # Kalman filters: reset per clip
        kalman_l = BallKalman()
        kalman_r = BallKalman()

        pre_roll_n = max(2, int(PRE_ROLL_S * args.fps))
        preroll_l: deque[np.ndarray] = deque(maxlen=pre_roll_n)
        preroll_r: deque[np.ndarray] = deque(maxlen=pre_roll_n)

        state = State.IDLE
        record_l: list[np.ndarray] = []
        record_r: list[np.ndarray] = []

        # live path for display during TRACKING
        live_path_l: dict[int, tuple[float, float]] = {}
        live_path_r: dict[int, tuple[float, float]] = {}

        last_det_time_l = -999.0
        last_det_time_r = -999.0
        frozen_l: np.ndarray | None = None
        frozen_r: np.ndarray | None = None

        # FPS measurement
        fps_frames = 0
        fps_t0 = time.monotonic()
        live_fps = args.fps

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1200, 520)

        while True:
            ok_l, frame_l = read_cam(cap_l)
            ok_r, frame_r = read_cam(cap_r)
            if not ok_l or frame_l is None:
                time.sleep(0.01)
                continue
            if not ok_r or frame_r is None:
                time.sleep(0.01)
                continue

            t_now = time.monotonic()
            fps_frames += 1
            if t_now - fps_t0 >= 3.0:
                live_fps = fps_frames / (t_now - fps_t0)
                fps_frames = 0
                fps_t0 = t_now

            # Motion always runs — keeps the background model continuously updated
            mr_l = motion_l.process(frame_l)
            mr_r = motion_r.process(frame_r)

            max_buf = max(MIN_TRACK_FRAMES + 1, int(MAX_BUFFER_S * live_fps))

            # ── IDLE ──────────────────────────────────────────────────────────
            if state == State.IDLE:
                preroll_l.append(frame_l.copy())
                preroll_r.append(frame_r.copy())

                det_l, det_r = detect_pair(
                    model, frame_l, frame_r, mr_l, mr_r,
                    kalman_l, kalman_r, args.conf, classes,
                )
                if det_l:
                    kalman_l.update(det_l[0], det_l[1])
                else:
                    kalman_l.predict()
                if det_r:
                    kalman_r.update(det_r[0], det_r[1])
                else:
                    kalman_r.predict()

                trigger = (
                    (det_l is not None and det_r is not None)
                    if require_both
                    else (det_l is not None or det_r is not None)
                )
                if trigger:
                    state = State.TRACKING
                    record_l = list(preroll_l)
                    record_r = list(preroll_r)
                    live_path_l = {}
                    live_path_r = {}
                    n_pre = len(preroll_l)
                    if det_l:
                        live_path_l[n_pre - 1] = (det_l[0], det_l[1])
                    if det_r:
                        live_path_r[n_pre - 1] = (det_r[0], det_r[1])
                    last_det_time_l = t_now if det_l else -999.0
                    last_det_time_r = t_now if det_r else -999.0
                    print(f"▶ TRACKING — preroll={n_pre} frames | fps≈{live_fps:.1f}")

                disp_l = frame_l.copy()
                disp_r = frame_r.copy()
                paint_det(disp_l, det_l, TRAIL_L)
                paint_det(disp_r, det_r, TRAIL_R)
                combo = make_panel(disp_l, disp_r)
                add_state_overlay(
                    combo, State.IDLE,
                    f"LOOKING FOR BALL  ({'both' if require_both else 'any'} camera)",
                )
                cv2.imshow(WINDOW, combo)

            # ── TRACKING ──────────────────────────────────────────────────────
            elif state == State.TRACKING:
                if len(record_l) < max_buf:
                    record_l.append(frame_l.copy())
                    record_r.append(frame_r.copy())
                frozen_l = frame_l.copy()
                frozen_r = frame_r.copy()

                det_l, det_r = detect_pair(
                    model, frame_l, frame_r, mr_l, mr_r,
                    kalman_l, kalman_r, args.conf, classes,
                )
                frame_idx = len(record_l) - 1
                if det_l:
                    kalman_l.update(det_l[0], det_l[1])
                    last_det_time_l = t_now
                    live_path_l[frame_idx] = (det_l[0], det_l[1])
                else:
                    kalman_l.predict()
                if det_r:
                    kalman_r.update(det_r[0], det_r[1])
                    last_det_time_r = t_now
                    live_path_r[frame_idx] = (det_r[0], det_r[1])
                else:
                    kalman_r.predict()

                both_lost = (
                    t_now - last_det_time_l > LOST_S
                    and t_now - last_det_time_r > LOST_S
                )
                buffer_full = len(record_l) >= max_buf
                if both_lost or buffer_full:
                    reason = "buffer full" if buffer_full else "ball lost"
                    if len(record_l) >= MIN_TRACK_FRAMES:
                        print(f"■ {reason} → PROCESSING ({len(record_l)} frames)")
                        state = State.PROCESSING
                    else:
                        print(f"■ {reason} — clip too short ({len(record_l)} frames), discarded")
                        state = State.IDLE
                        record_l.clear()
                        record_r.clear()
                        live_path_l.clear()
                        live_path_r.clear()
                        kalman_l.reset()
                        kalman_r.reset()

                # Display live growing trail
                disp_l = frame_l.copy()
                disp_r = frame_r.copy()
                draw_trail(disp_l, live_path_l, TRAIL_L, up_to=frame_idx)
                draw_trail(disp_r, live_path_r, TRAIL_R, up_to=frame_idx)
                paint_det(disp_l, det_l, TRAIL_L)
                paint_det(disp_r, det_r, TRAIL_R)
                combo = make_panel(disp_l, disp_r)
                add_state_overlay(combo, State.TRACKING, f"● REC  {len(record_l)} frames")
                cv2.imshow(WINDOW, combo)

            # ── PROCESSING ────────────────────────────────────────────────────
            elif state == State.PROCESSING:
                # Freeze the last tracked frame as the background for progress display
                bg_l = frozen_l if frozen_l is not None else frame_l
                bg_r = frozen_r if frozen_r is not None else frame_r

                def show_proc(i: int, n: int) -> None:
                    pct = int(100 * i / max(n, 1))
                    p = make_panel(bg_l.copy(), bg_r.copy())
                    add_state_overlay(p, State.PROCESSING, f"PROCESSING…  {pct}%")
                    cv2.imshow(WINDOW, p)
                    cv2.waitKey(1)

                show_proc(0, 1)

                # Re-run YOLO on every recorded frame to build the complete path
                path_l, path_r = build_paths(
                    record_l, record_r, model, classes, on_progress=show_proc,
                )
                path_l = interpolate_path(path_l, len(record_l))
                path_r = interpolate_path(path_r, len(record_r))
                print(f"  Path pts: left={len(path_l)}  right={len(path_r)}")

                if path_l or path_r:
                    try:
                        save_clip(record_l, record_r, path_l, path_r, live_fps)
                    except Exception as exc:
                        print(f"  Save error: {exc}")
                else:
                    print("  No ball found in backfill — clip discarded")

                # Reset clip state; motion stays warm
                record_l.clear()
                record_r.clear()
                live_path_l.clear()
                live_path_r.clear()
                kalman_l.reset()
                kalman_r.reset()
                preroll_l.clear()
                preroll_r.clear()
                frozen_l = frozen_r = None
                state = State.IDLE
                print("● IDLE — looking for ball again\n")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        if cap_l is not None:
            cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
