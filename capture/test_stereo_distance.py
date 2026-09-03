"""
Dual-camera distance test — still football, any indoor environment.

Uses the standard COCO-pretrained YOLOv8n (class 32 = sports ball).
No outdoor-trained custom model needed; downloads ~6 MB automatically on first run.

Depth (metres) from parallel stereo:
  Z = (focal_px * baseline_m) / disparity_px

Calibrate once (C key) at a known distance, then it saves focal_px for future runs.

Usage:
  python capture/test_stereo_distance.py --list               # find camera indices
  python capture/test_stereo_distance.py --left 0 --right 1
  python capture/test_stereo_distance.py --left 0 --right 1 --baseline 1.0 --known-distance 5.0

Keys: C = calibrate | Q = quit
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = PROJECT_ROOT / "exports" / "stereo" / "simple_focal.json"

# COCO pretrained — class 32 = sports ball, class 0 = person
# yolov8m (medium) is noticeably better than small at finding balls at a distance.
COCO_MODEL = "yolov8m.pt"
SPORTS_BALL_CLASS = 32
PERSON_CLASS = 0

IMAGE_SIZE = 640
CONFIDENCE = 0.15          # low — restricted to sports-ball class so false positives are rare

# Hardcoded focal length for this specific camera+lens+resolution combination.
# Measured by calibration: Arducam UC-844, native 640×480, rotated -90° → 480×640.
# Physical equivalent: ~2.5 mm focal length on a 1/4" sensor.
# If you change cameras, resolution, or lenses, press C at a known distance to recalibrate.
HARDCODED_FOCAL_PX = 438.4   # px  (baseline 1.0 m, calibrated at 5.0 m, disparity 87.7 px)

# SimpleBlobDetector fallback — finds compact, round, convex blobs only.
# Windows / furniture / heads fail the circularity + convexity + inertia tests.
BLOB_MIN_AREA = 300        # px² — ignore tiny specks
BLOB_MAX_AREA = 120_000    # px² — ignore huge blobs
BLOB_MIN_CIRCULARITY = 0.55  # 0–1; circle=1, square≈0.78, window frame≈0.1
BLOB_MIN_CONVEXITY = 0.80
BLOB_MIN_INERTIA = 0.40    # 0=line, 1=circle; elongated shapes (windows) fail this
BASELINE_M = 1.0
KNOWN_DISTANCE_M = 5.0
SMOOTH_N = 8
WINDOW = "Hawkeye Stereo Distance — C calibrate | Q quit"


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
    """
    Open a USB camera at its native (maximum) resolution.

    We do NOT request a specific width/height here — letting the driver use
    its default gives the full sensor FOV. Requesting e.g. 1280×720 can
    cause some cameras to apply digital crop / letterbox instead.
    """
    last_err = f"Could not open camera {index}."
    for name, backend in camera_backends():
        print(f"  Trying camera {index} via {name}...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Disable any hardware AE that might darken frames while settling
        try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Keep internal buffer small so frames are always fresh
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)

        # Flush a few frames — some cameras output black/green on first read
        ok, frame = False, None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None and frame.any():
                break
            time.sleep(0.05)

        if ok and frame is not None and frame.any():
            h, w = frame.shape[:2]
            print(f"  Camera {index} OK via {name} — native {w}x{h}")
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
# Ball detection (COCO)
# ---------------------------------------------------------------------------

def load_coco_model() -> YOLO:
    """Load COCO YOLOv8s. Downloads ~22 MB on first run."""
    print(f"Loading COCO model ({COCO_MODEL}) — downloads automatically if not cached...")
    model = YOLO(COCO_MODEL)
    print("Model ready.")
    return model


def _yolo_predict(model: YOLO, frame, conf: float, classes: list[int]):
    """Run YOLO and return the raw boxes result (or None)."""
    results = model.predict(
        frame,
        conf=conf,
        classes=classes,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    return results[0].boxes


def _detect_yolo_ball(
    model: YOLO,
    frame,
    conf: float,
) -> tuple[float, float, float, float, float, float, float] | None:
    """YOLO sports-ball detection only. Returns (cx,cy,x1,y1,x2,y2,conf) or None."""
    boxes = _yolo_predict(model, frame, conf, [SPORTS_BALL_CLASS])
    if boxes is None:
        return None
    best = None
    best_conf = -1.0
    for box in boxes:
        c = float(box.conf[0])
        if c <= best_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_conf = c
        best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2, c)
    return best


def _make_blob_detector() -> cv2.SimpleBlobDetector:
    """
    Build a SimpleBlobDetector tuned for a round ball.

    A ball scores high on circularity (~0.85+), convexity (~0.9+) and inertia
    (~0.7+).  Windows, door frames, heads, and light fittings all fail at
    least one of these tests.
    """
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = BLOB_MIN_AREA
    p.maxArea = BLOB_MAX_AREA
    p.filterByCircularity = True
    p.minCircularity = BLOB_MIN_CIRCULARITY
    p.filterByConvexity = True
    p.minConvexity = BLOB_MIN_CONVEXITY
    p.filterByInertia = True
    p.minInertiaRatio = BLOB_MIN_INERTIA
    p.filterByColor = False
    p.minDistBetweenBlobs = 20
    return cv2.SimpleBlobDetector_create(p)


# Create once at module level so it isn't rebuilt every frame
_BLOB_DETECTOR = _make_blob_detector()


def _person_boxes(model: YOLO, frame) -> list[tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) for every person YOLO finds."""
    boxes = _yolo_predict(model, frame, conf=0.35, classes=[PERSON_CLASS])
    if boxes is None:
        return []
    return [tuple(int(v) for v in box.xyxy[0].tolist()) for box in boxes]


def _detect_blob(
    frame,
    person_boxes: list[tuple[int, int, int, int]],
) -> tuple[float, float, float, float, float, float, float] | None:
    """
    SimpleBlobDetector fallback.

    Steps:
      1. Convert to grayscale and blur lightly.
      2. Mask out every person bounding box so body parts can't fire.
      3. Run the detector — it only returns compact, round, convex blobs.
      4. Return the largest surviving blob (most likely the ball).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mask person regions before detection
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for (px1, py1, px2, py2) in person_boxes:
        pad = 15
        y1c = max(0, py1 - pad)
        y2c = min(gray.shape[0], py2 + pad)
        x1c = max(0, px1 - pad)
        x2c = min(gray.shape[1], px2 + pad)
        mask[y1c:y2c, x1c:x2c] = 0

    # Blob detector finds dark blobs by default; try both polarities
    masked = cv2.bitwise_and(gray, mask)
    blurred = cv2.GaussianBlur(masked, (7, 7), 0)

    keypoints = _BLOB_DETECTOR.detect(blurred)

    # Also try inverted (light ball on dark background)
    inv = cv2.bitwise_not(blurred)
    keypoints_inv = _BLOB_DETECTOR.detect(inv)

    # Merge and pick the largest
    all_kp = list(keypoints) + list(keypoints_inv)
    if not all_kp:
        return None

    best_kp = max(all_kp, key=lambda kp: kp.size)
    cx, cy = best_kp.pt
    r = best_kp.size / 2.0
    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
    return (cx, cy, x1, y1, x2, y2, 0.50)


def detect_ball(
    model: YOLO,
    frame,
    conf: float,
) -> tuple[float, float, float, float, float, float, float] | None:
    """
    1. Try YOLO sports-ball (class 32, low confidence).
    2. If nothing found, detect persons to build an exclusion mask.
    3. Run SimpleBlobDetector on the masked frame — only compact round blobs pass.
    """
    det = _detect_yolo_ball(model, frame, conf)
    if det is not None:
        return det
    persons = _person_boxes(model, frame)
    return _detect_blob(frame, persons)


# ---------------------------------------------------------------------------
# Stereo maths
# ---------------------------------------------------------------------------

def default_focal_px(image_width: int) -> float:
    """Rough guess assuming ~70° HFOV. Press C at a known distance to calibrate."""
    return (max(image_width, 1) / 2.0) / math.tan(math.radians(70.0 / 2.0))


def horizontal_disparity(cx_l: float, w_l: int, cx_r: float, w_r: int) -> float:
    """Centre-relative disparity — correct even when the two cameras have different resolutions."""
    return (cx_l - w_l / 2.0) - (cx_r - w_r / 2.0)


def depth_from_disparity(disparity_px: float, focal_px: float, baseline_m: float) -> float | None:
    if disparity_px <= 1.0 or focal_px <= 0 or baseline_m <= 0:
        return None
    return (focal_px * baseline_m) / disparity_px


def focal_from_known(disparity_px: float, known_z_m: float, baseline_m: float) -> float | None:
    if disparity_px <= 1.0 or known_z_m <= 0 or baseline_m <= 0:
        return None
    return (known_z_m * disparity_px) / baseline_m


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------

def load_saved_calib() -> dict | None:
    if not CALIB_PATH.exists():
        return None
    try:
        data = json.loads(CALIB_PATH.read_text())
        if float(data["focal_px"]) <= 0:
            return None
        return data
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_calib(focal_px: float, baseline_m: float, known_z_m: float,
               disparity_px: float, image_width: int) -> None:
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps({
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "calibrated_at_m": known_z_m,
        "disparity_px": disparity_px,
        "image_width": image_width,
        "note": "Redo C if cameras move, resolution changes, or baseline changes",
    }, indent=2))
    print(f"Saved focal_px={focal_px:.1f} → {CALIB_PATH}")


def resolve_focal(cli_focal: float | None, baseline_m: float, image_width: int) -> float:
    if cli_focal is not None:
        if cli_focal <= 0:
            raise ValueError("--focal must be > 0")
        print(f"Using CLI focal_px={cli_focal:.1f}")
        return cli_focal

    saved = load_saved_calib()
    if saved is not None:
        focal = float(saved["focal_px"])
        saved_w = int(saved.get("image_width") or image_width)
        if saved_w > 0 and saved_w != image_width:
            focal *= image_width / saved_w
            print(f"Scaled saved focal for resolution change → focal_px={focal:.1f}")
        saved_b = float(saved.get("baseline_m") or baseline_m)
        if abs(saved_b - baseline_m) > 0.01:
            print(
                f"WARNING: saved calib used baseline {saved_b:.3f} m, "
                f"now running {baseline_m:.3f} m — press C to recalibrate."
            )
        print(f"Using saved focal_px={focal:.1f}")
        return focal

    # Fall back to the hardcoded value from our own calibration run.
    # This is correct for Arducam UC-844 at 640×480 rotated -90° (→ 480×640).
    # Scale it if the actual image width differs from the calibration width.
    CALIB_WIDTH = 480  # width used when HARDCODED_FOCAL_PX was measured
    focal = HARDCODED_FOCAL_PX
    if image_width != CALIB_WIDTH and CALIB_WIDTH > 0:
        focal = focal * image_width / CALIB_WIDTH
        print(
            f"Using hardcoded focal_px={HARDCODED_FOCAL_PX:.1f} scaled "
            f"for width {CALIB_WIDTH}→{image_width} → focal_px={focal:.1f}"
        )
    else:
        print(
            f"Using hardcoded focal_px={focal:.1f} "
            "(Arducam UC-844, 480px wide after -90° rotation). "
            "Press C at a known distance to recalibrate if needed."
        )
    return focal


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def paint(frame, title: str,
          det: tuple[float, float, float, float, float, float, float] | None) -> None:
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    if det is None:
        cv2.putText(frame, "no ball", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return
    cx, cy, x1, y1, x2, y2, conf = det
    # Hough detections have conf==0.50 exactly; YOLO detections vary
    method = "hough" if conf == 0.50 else "yolo"
    colour = (255, 160, 0) if method == "hough" else (0, 165, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
    cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), -1)
    cv2.putText(
        frame, f"ball [{method}] {conf:.2f}", (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
    )


def add_status_bar(combo, status: str) -> None:
    h = combo.shape[0]
    cv2.rectangle(combo, (0, h - 44), (combo.shape[1], h), (0, 0, 0), -1)
    cv2.putText(combo, status, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rotate_frame(frame, degrees: int):
    """Rotate frame so the wide sensor axis is horizontal (cam mounted sideways)."""
    if degrees == 0:
        return frame
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees in (-90, 270):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"Unsupported rotation {degrees}; use 0, 90, -90, or 180")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual-camera football distance test",
        # Allow --rotate=-90 with equals sign (argparse treats bare '-90' as a flag)
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--left",  type=int, default=0, help="Left camera index (default 0)")
    parser.add_argument("--right", type=int, default=1, help="Right camera index (default 1)")
    parser.add_argument("--baseline", type=float, default=BASELINE_M,
                        help=f"Camera separation in metres (default {BASELINE_M})")
    parser.add_argument("--known-distance", type=float, default=KNOWN_DISTANCE_M,
                        help=f"True distance when pressing C (default {KNOWN_DISTANCE_M})")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length in pixels (skips saved calib)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Detection confidence (default {CONFIDENCE})")
    parser.add_argument("--list", action="store_true",
                        help="Probe camera indices 0–5 and exit")
    parser.add_argument(
        "--rotate", type=int, default=-90,
        help=(
            "Rotate each frame so the wide FOV axis is horizontal. "
            "Default 90 (clockwise). Use --rotate=-90 for counter-clockwise, "
            "--rotate=0 if already landscape, --rotate=180 to flip."
        ),
    )
    # Parse with a small trick so '-90' is not mistaken for a flag:
    # users must write --rotate=-90 (with '=').  We also handle the bare
    # negative case gracefully by pre-processing sys.argv.
    raw_args = sys.argv[1:]
    fixed_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        # Turn '--rotate -90' (two tokens) into '--rotate=-90' (one token)
        if arg == "--rotate" and i + 1 < len(raw_args) and raw_args[i + 1].lstrip("-").isdigit():
            fixed_args.append(f"--rotate={raw_args[i + 1]}")
            i += 2
        else:
            fixed_args.append(arg)
            i += 1
    args = parser.parse_args(fixed_args)

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
    if args.rotate not in (0, 90, -90, 180, 270):
        print("ERROR: --rotate must be 0, 90, -90, or 180", file=sys.stderr)
        return 1

    print(
        f"\nBaseline={args.baseline:.3f} m | left cam={args.left} | right cam={args.right}\n"
        f"Known-distance for C key = {args.known_distance:.2f} m | conf={args.conf:.2f}\n"
        f"Rotation = {args.rotate}°  (use --rotate=0 if already landscape)\n"
        "Q = quit\n"
    )

    model = load_coco_model()

    print(f"\nOpening left camera (index {args.left})...")
    cap_l = None
    cap_r = None
    try:
        cap_l = open_camera(args.left)
        # Small delay — on macOS, opening two AVFoundation devices back-to-back
        # can starve the second camera of frames.
        time.sleep(0.5)
        print(f"Opening right camera (index {args.right})...")
        try:
            cap_r = open_camera(args.right)
        except RuntimeError:
            cap_l.release()
            raise

        # Probe actual frame size (don't assume request was honoured)
        ok_l, probe_l = read_frame(cap_l, retries=10)
        ok_r, probe_r = read_frame(cap_r, retries=10)
        if not ok_l or probe_l is None:
            print("ERROR: left camera gave no frames.", file=sys.stderr)
            return 1
        if not ok_r or probe_r is None:
            print("ERROR: right camera gave no frames.", file=sys.stderr)
            return 1

        probe_l = rotate_frame(probe_l, args.rotate)
        probe_r = rotate_frame(probe_r, args.rotate)
        w_l, h_l = probe_l.shape[1], probe_l.shape[0]
        w_r, h_r = probe_r.shape[1], probe_r.shape[0]
        print(f"Left  resolution after rotation: {w_l}x{h_l}")
        print(f"Right resolution after rotation: {w_r}x{h_r}")
        if w_l != w_r or h_l != h_r:
            print("WARNING: resolutions differ — using centre-relative disparity (still valid).")

        try:
            focal_px = resolve_focal(args.focal, args.baseline, w_l)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        depths: list[float] = []
        panel_w = min(1600, w_l + w_r)
        panel_h = min(900, max(h_l, h_r))
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, panel_w, panel_h)

        while True:
            ok_l, frame_l = read_frame(cap_l, retries=5)
            ok_r, frame_r = read_frame(cap_r, retries=5)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                print("Frame grab failed — check USB connections.")
                time.sleep(0.05)
                continue

            frame_l = rotate_frame(frame_l, args.rotate)
            frame_r = rotate_frame(frame_r, args.rotate)

            det_l = detect_ball(model, frame_l, args.conf)
            det_r = detect_ball(model, frame_r, args.conf)

            paint(frame_l, f"LEFT  cam {args.left}  {frame_l.shape[1]}x{frame_l.shape[0]}", det_l)
            paint(frame_r, f"RIGHT cam {args.right}  {frame_r.shape[1]}x{frame_r.shape[0]}", det_r)

            status = "need ball in BOTH views"

            if det_l is not None and det_r is not None:
                disparity = horizontal_disparity(det_l[0], frame_l.shape[1],
                                                 det_r[0], frame_r.shape[1])
                if disparity <= 1.0:
                    depths.clear()
                    status = (
                        f"bad disparity={disparity:.1f}px — "
                        "try swapping --left/--right, or check both cameras aim the same direction"
                    )
                else:
                    z = depth_from_disparity(disparity, focal_px, args.baseline)
                    if z is not None:
                        depths.append(z)
                        if len(depths) > SMOOTH_N:
                            depths.pop(0)
                        z_s = float(np.median(depths))
                        status = (
                            f"distance ≈ {z_s:.2f} m   "
                            f"(raw {z:.2f} m | disp {disparity:.1f} px | f={focal_px:.0f} px)"
                        )
            else:
                depths.clear()

            # Resize both to the same height before hstack
            target_h = min(frame_l.shape[0], frame_r.shape[0])
            def fit_h(img, th):
                if img.shape[0] == th:
                    return img
                scale = th / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), th), interpolation=cv2.INTER_AREA)

            combo = np.hstack([fit_h(frame_l, target_h), fit_h(frame_r, target_h)])
            add_status_bar(combo, status)
            cv2.imshow(WINDOW, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("c"), ord("C")):
                if det_l is None or det_r is None:
                    print("Calibrate: ball not visible in both cameras — move it into both views.")
                    continue
                disp = horizontal_disparity(det_l[0], frame_l.shape[1],
                                            det_r[0], frame_r.shape[1])
                if disp <= 1.0:
                    print(
                        f"Calibrate failed: disparity={disp:.1f} px. "
                        "Try swapping --left/--right."
                    )
                    continue
                new_f = focal_from_known(disp, args.known_distance, args.baseline)
                if new_f is None:
                    print("Calibrate failed: bad numbers.")
                    continue
                focal_px = new_f
                depths.clear()
                save_calib(focal_px, args.baseline, args.known_distance,
                           disp, frame_l.shape[1])
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
