"""
Dual-camera stereo distance & between-posts detection test.

Detection model:
  - Default: COCO pretrained yolov8n.pt (class 32 = sports ball) — works anywhere indoors.
  - For outdoor/sky use: pass --model path/to/football_yolov8n.pt (no --classes flag needed).

Stereo geometry:
  Both cameras must see the ball at the same time.
  The 3D X position from triangulation tells you if it was between the posts.
  "Both cameras see it" alone is NOT enough — the ball can be outside the posts and
  still be visible in both camera fields of view.

Plane-crossing verdict:
  BETWEEN POSTS : ball X within [0, baseline_m] at crossing moment
  OUTSIDE POSTS : ball X outside that range

Detection range note:
  Runs YOLO at imgsz=640 (not 320) so the ball is detectable from ~1 m to ~20 m.
  Both frames are batched into a single model.predict() call for efficiency.

Usage:
  python capture/test_stereo_distance.py --list
  python capture/test_stereo_distance.py --left 0 --right 1
  python capture/test_stereo_distance.py --left 0 --right 1 --baseline 1.0 --known-distance 5.0
  python capture/test_stereo_distance.py --left 0 --right 1 --model models/football_yolov8n.pt

Keys: C = calibrate focal length at --known-distance | Q = quit
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from auto_exposure import try_set
import stereo_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NOTE: test_stereo_distance.py assumes PARALLEL (forward-facing) cameras.
# For the real converging rig (25° inward, 25° upward) use stereo_calibrate.py.

COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32        # only used with the COCO model

IMAGE_SIZE = 640              # keep at 640 — ball is ~2-3 px at 20 m, too small at 320
CONFIDENCE = 0.30
SMOOTH_N = 8
WINDOW = "Hawkeye Stereo — C calibrate | Q quit"


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def camera_backends() -> list[tuple[str, int]]:
    system = platform.system()
    if system == "Darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("default", cv2.CAP_ANY)]
    if system == "Linux":
        return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
    return [("MSMF", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera at its native resolution (no resize request = full FOV)."""
    last_err = f"Could not open camera {index}."
    for name, backend in camera_backends():
        print(f"  Trying camera {index} via {name}...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)

        # Flush until we get a real (non-black) frame
        ok, frame = False, None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None and frame.any():
                break
            time.sleep(0.05)

        if ok and frame is not None and frame.any():
            h, w = frame.shape[:2]
            print(f"  Camera {index} OK via {name} — {w}x{h}")
            return cap

        cap.release()
        last_err = (
            f"Camera {index} opened via {name} but gave no valid frames. "
            "Close other camera apps, unplug/replug, then retry."
        )

    raise RuntimeError(last_err)


def read_frame(cap: cv2.VideoCapture, retries: int = 5):
    for _ in range(max(retries, 1)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.02)
    return False, None


# ---------------------------------------------------------------------------
# Ball detection — batched (both frames in one call)
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_coco: bool) -> tuple[YOLO, list[int] | None]:
    """
    Returns (model, classes_filter).
    classes_filter is [32] for COCO (sports ball only), None for custom model (all classes).
    """
    print(f"Loading model: {model_path} ...")
    model = YOLO(model_path)
    if use_coco:
        print("COCO model — filtering for class 32 (sports ball).")
        return model, [SPORTS_BALL_CLASS]
    print("Custom model — using all classes (no class filter).")
    return model, None


def detect_both(
    model: YOLO,
    frame_l,
    frame_r,
    conf: float,
    classes: list[int] | None,
) -> tuple[
    tuple[float, float, float, float, float, float, float] | None,
    tuple[float, float, float, float, float, float, float] | None,
]:
    """
    Run one batched YOLO inference on both frames simultaneously.
    Returns (det_left, det_right) where each is (cx, cy, x1, y1, x2, y2, conf) or None.

    Batching is ~1.6-1.9× faster than two separate calls because the GPU/CPU
    overhead is paid once, not twice.
    """
    predict_kwargs: dict = dict(conf=conf, imgsz=IMAGE_SIZE, verbose=False)
    if classes is not None:
        predict_kwargs["classes"] = classes

    results = model.predict([frame_l, frame_r], **predict_kwargs)
    return _parse_best(results[0]), _parse_best(results[1])


def _parse_best(result) -> tuple[float, float, float, float, float, float, float] | None:
    """Extract the highest-confidence box from a single result."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best = None
    best_conf = -1.0
    for box in result.boxes:
        c = float(box.conf[0])
        if c <= best_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_conf = c
        best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2, c)
    return best


# ---------------------------------------------------------------------------
# Stereo maths
# ---------------------------------------------------------------------------

def default_focal_px(image_width: int) -> float:
    """Spec-based: 70° HFOV (B0332 + LN013). Press C at a known distance to refine."""
    f, _ = cfg.get_focal_px(image_width)
    return f


def horizontal_disparity(cx_l: float, w_l: int, cx_r: float, w_r: int) -> float:
    """
    Centre-relative disparity.
    Correct for parallel cameras even when the two cameras have different resolutions.
    """
    return (cx_l - w_l / 2.0) - (cx_r - w_r / 2.0)


def depth_from_disparity(disparity_px: float, focal_px: float, baseline_m: float) -> float | None:
    if disparity_px <= 1.0 or focal_px <= 0 or baseline_m <= 0:
        return None
    return (focal_px * baseline_m) / disparity_px


def focal_from_known(disparity_px: float, known_z_m: float, baseline_m: float) -> float | None:
    if disparity_px <= 1.0 or known_z_m <= 0 or baseline_m <= 0:
        return None
    return (known_z_m * disparity_px) / baseline_m


def ball_x_position(
    cx_l: float,
    w_l: int,
    depth_m: float,
    focal_px: float,
) -> float:
    """
    Estimate the ball's lateral (X) position in metres relative to the LEFT camera.

    With the left camera at X=0 and right camera at X=baseline_m, a result in
    [0, baseline_m] means the ball is between the posts.

    Formula: X = (cx_l - w_l/2) * depth_m / focal_px
    """
    return (cx_l - w_l / 2.0) * depth_m / focal_px


def between_posts_verdict(
    x_m: float,
    baseline_m: float,
    margin_m: float = 0.10,
) -> str:
    """
    Returns a verdict string based on the ball's X position.
    margin_m: allow small measurement error near the post edges.
    """
    if -margin_m <= x_m <= baseline_m + margin_m:
        return f"BETWEEN POSTS  x={x_m:.2f} m"
    if x_m < 0:
        return f"OUTSIDE (left)  x={x_m:.2f} m"
    return f"OUTSIDE (right)  x={x_m:.2f} m"


# ---------------------------------------------------------------------------
# Calibration persistence — delegates to stereo_config
# ---------------------------------------------------------------------------

def resolve_focal(cli_focal: float | None, baseline_m: float, image_width: int) -> float:
    if cli_focal is not None:
        if cli_focal <= 0:
            raise ValueError("--focal must be > 0")
        print(f"Using CLI focal_px={cli_focal:.1f}")
        return cli_focal

    focal, src = cfg.get_focal_px(image_width)
    saved_b_raw = None
    # Warn if the saved baseline doesn't match the current one
    for path in (cfg.CALIB_PATH, cfg.OLD_CALIB_PATH):
        if path.exists():
            try:
                saved_b_raw = float(json.loads(path.read_text()).get("baseline_m", baseline_m))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            break
    if saved_b_raw is not None and abs(saved_b_raw - baseline_m) > 0.01:
        print(
            f"WARNING: saved calib used baseline {saved_b_raw:.3f} m, "
            f"now running {baseline_m:.3f} m — press C to recalibrate."
        )
    print(f"focal_px={focal:.1f}  ← {src}")
    return focal


def save_calib(focal_px: float, baseline_m: float, known_z_m: float,
               disparity_px: float, image_width: int) -> None:
    cfg.save_calib(
        focal_px=focal_px,
        baseline_m=baseline_m,
        image_width=image_width,
        image_height=cfg.NATIVE_HEIGHT,
        calibrated_at_height_m=known_z_m,
    )
    print(f"Saved focal_px={focal_px:.1f}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def paint(frame, title: str,
          det: tuple[float, float, float, float, float, float, float] | None) -> None:
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (0, 255, 255), 2, cv2.LINE_AA)
    if det is None:
        cv2.putText(frame, "no ball", (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (0, 0, 255), 2, cv2.LINE_AA)
        return
    cx, cy, x1, y1, x2, y2, conf = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), -1)
    cv2.putText(frame, f"ball {conf:.2f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2, cv2.LINE_AA)


def status_bar(combo, line1: str, line2: str = "") -> None:
    h = combo.shape[0]
    bar_h = 44 if not line2 else 68
    cv2.rectangle(combo, (0, h - bar_h), (combo.shape[1], h), (0, 0, 0), -1)
    cv2.putText(combo, line1, (12, h - bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (0, 255, 255), 2, cv2.LINE_AA)
    if line2:
        color = (0, 255, 80) if "BETWEEN" in line2 else (0, 80, 255)
        cv2.putText(combo, line2, (12, h - bar_h + 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, color, 2, cv2.LINE_AA)


def fit_height(img, target_h: int):
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-camera stereo distance test")
    parser.add_argument("--left",  type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--baseline", type=float, default=cfg.BASELINE_M,
                        help=f"Camera separation in metres (default {cfg.BASELINE_M})")
    parser.add_argument("--known-distance", type=float, default=5.0,
                        help="True distance when pressing C to calibrate (default 5.0)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length in pixels")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Detection confidence (default {CONFIDENCE})")
    parser.add_argument("--model", type=str, default=COCO_MODEL,
                        help=f"Model path (default: {COCO_MODEL} — COCO sports ball). "
                             "Use models/football_yolov8n.pt for custom model.")
    parser.add_argument("--list", action="store_true",
                        help="Probe camera indices 0–5 and exit")
    args = parser.parse_args()

    if args.list:
        print("Probing cameras 0–5:")
        for i in range(6):
            try:
                cap = open_camera(i)
                print(f"  → index {i} OK")
                cap.release()
                time.sleep(0.3)
            except RuntimeError as exc:
                print(f"  → index {i} FAIL: {exc}")
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right must be different indices", file=sys.stderr)
        return 1
    if args.baseline <= 0:
        print("ERROR: --baseline must be > 0", file=sys.stderr)
        return 1
    if args.known_distance <= 0:
        print("ERROR: --known-distance must be > 0", file=sys.stderr)
        return 1
    if not 0.0 < args.conf <= 1.0:
        print("ERROR: --conf must be in (0, 1]", file=sys.stderr)
        return 1

    # Decide whether this is the COCO model or a custom one
    model_path = args.model
    use_coco = (model_path == COCO_MODEL)
    if not use_coco:
        full_path = PROJECT_ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
        if not full_path.exists():
            print(f"ERROR: model not found: {full_path}", file=sys.stderr)
            return 1
        model_path = str(full_path)

    print(
        f"\nBaseline = {args.baseline:.3f} m  |  left={args.left}  right={args.right}\n"
        f"Known distance for C = {args.known_distance:.2f} m  |  conf={args.conf:.2f}\n"
        f"Detection range: ~1 m – 20 m (imgsz={IMAGE_SIZE})\n"
        "Q = quit\n"
    )

    model, classes = load_model(model_path, use_coco)

    cap_l = cap_r = None
    try:
        print(f"\nOpening left camera (index {args.left})...")
        cap_l = open_camera(args.left)
        time.sleep(0.5)   # give AVFoundation time between two devices on macOS
        print(f"Opening right camera (index {args.right})...")
        cap_r = open_camera(args.right)  # finally block handles cleanup on failure

        ok_l, probe_l = read_frame(cap_l, retries=10)
        ok_r, probe_r = read_frame(cap_r, retries=10)
        if not ok_l or probe_l is None:
            print("ERROR: left camera gave no frames.", file=sys.stderr)
            return 1
        if not ok_r or probe_r is None:
            print("ERROR: right camera gave no frames.", file=sys.stderr)
            return 1

        w_l, h_l = probe_l.shape[1], probe_l.shape[0]
        w_r, h_r = probe_r.shape[1], probe_r.shape[0]
        print(f"Left  native: {w_l}x{h_l}  |  Right native: {w_r}x{h_r}")
        if w_l != w_r or h_l != h_r:
            print("WARNING: different resolutions — centre-relative disparity still valid.")

        try:
            focal_px = resolve_focal(args.focal, args.baseline, w_l)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        depths: list[float] = []
        # Window height matches what we actually display: min of the two heights
        # (the status bar is drawn inside the combo image, not as extra space)
        display_h = min(h_l, h_r)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(1600, w_l + w_r), min(900, display_h))

        while True:
            ok_l, frame_l = read_frame(cap_l, retries=5)
            ok_r, frame_r = read_frame(cap_r, retries=5)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                print("Frame grab failed — check USB connections.")
                time.sleep(0.05)
                continue

            # ---- single batched inference ----
            det_l, det_r = detect_both(model, frame_l, frame_r, args.conf, classes)

            paint(frame_l, f"LEFT  {frame_l.shape[1]}x{frame_l.shape[0]}", det_l)
            paint(frame_r, f"RIGHT {frame_r.shape[1]}x{frame_r.shape[0]}", det_r)

            dist_line = "need ball in BOTH views"
            verdict_line = ""

            if det_l is not None and det_r is not None:
                disparity = horizontal_disparity(
                    det_l[0], frame_l.shape[1],
                    det_r[0], frame_r.shape[1],
                )
                if disparity <= 1.0:
                    depths.clear()
                    dist_line = (
                        f"bad disparity={disparity:.1f} px — "
                        "try swapping --left/--right or check both cameras face same direction"
                    )
                else:
                    z = depth_from_disparity(disparity, focal_px, args.baseline)
                    if z is not None:
                        depths.append(z)
                        if len(depths) > SMOOTH_N:
                            depths.pop(0)
                        z_s = float(np.median(depths))

                        # X position in metres relative to the left camera
                        x_m = ball_x_position(det_l[0], frame_l.shape[1], z_s, focal_px)

                        dist_line = (
                            f"depth ≈ {z_s:.2f} m  "
                            f"(raw {z:.2f} m | disp {disparity:.1f} px | f={focal_px:.0f})"
                        )
                        verdict_line = between_posts_verdict(x_m, args.baseline)
            else:
                depths.clear()

            target_h = min(frame_l.shape[0], frame_r.shape[0])
            combo = np.hstack([fit_height(frame_l, target_h), fit_height(frame_r, target_h)])
            status_bar(combo, dist_line, verdict_line)
            cv2.imshow(WINDOW, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("c"), ord("C")):
                if det_l is None or det_r is None:
                    print("Calibrate: ball not visible in both cameras.")
                    continue
                disp = horizontal_disparity(
                    det_l[0], frame_l.shape[1],
                    det_r[0], frame_r.shape[1],
                )
                if disp <= 1.0:
                    print(f"Calibrate failed: disparity={disp:.1f} px. Try swapping --left/--right.")
                    continue
                new_f = focal_from_known(disp, args.known_distance, args.baseline)
                if new_f is None:
                    print("Calibrate failed: bad numbers.")
                    continue
                focal_px = new_f
                depths.clear()
                save_calib(focal_px, args.baseline, args.known_distance, disp, frame_l.shape[1])
                print(
                    f"Calibrated at {args.known_distance:.2f} m → "
                    f"focal_px={focal_px:.1f} px (disparity={disp:.1f} px)"
                )

    finally:
        if cap_l is not None:
            cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
