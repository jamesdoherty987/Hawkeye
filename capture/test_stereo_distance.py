"""
Simple dual-camera distance test (still football, indoor).

Assumes:
  - Two cameras on a rigid baseline (default 1.0 m apart)
  - Roughly parallel, both looking at the ball
  - Left camera = --left, right camera = --right

Depth (metres) from parallel stereo:
  Z = (focal_px * baseline_m) / disparity_px

Focal length is unknown until you calibrate once:
  1. Put the ball at a known distance (e.g. 5 m)
  2. Press C when both cams see it
  3. Script saves focal_px for next runs

Example:
  python capture/test_stereo_distance.py --left 0 --right 1
  python capture/test_stereo_distance.py --left 0 --right 1 --baseline 1.0
  python capture/test_stereo_distance.py --left 0 --right 1 --focal 900

Keys: C = calibrate at --known-distance | Q = quit
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

from auto_exposure import read_frame, try_set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "football_yolov8n.pt"
CALIB_PATH = PROJECT_ROOT / "exports" / "stereo" / "simple_focal.json"

IMAGE_SIZE = 640
CONFIDENCE = 0.45
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BASELINE_M = 1.0
KNOWN_DISTANCE_M = 5.0
SMOOTH_N = 8
WINDOW = "Hawkeye Stereo Distance — C calibrate | Q quit"


def default_focal_px(image_width: int) -> float:
    """Rough guess ≈70° HFOV — replace by pressing C at a known distance."""
    width = max(image_width, 1)
    return (width / 2.0) / math.tan(math.radians(70.0 / 2.0))


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


def open_camera(index: int) -> cv2.VideoCapture:
    for name, backend in camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = read_frame(cap, retries=15)
        if ok and frame is not None:
            print(
                f"Camera {index} opened via {name} "
                f"({frame.shape[1]}x{frame.shape[0]})"
            )
            return cap
        cap.release()
    raise RuntimeError(
        f"Couldn't open camera {index}. Try other --left / --right indices."
    )


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    print(f"Loading {MODEL_PATH}...")
    return YOLO(str(MODEL_PATH))


def depth_from_disparity(
    disparity_px: float,
    focal_px: float,
    baseline_m: float,
) -> float | None:
    if disparity_px <= 1.0 or focal_px <= 0 or baseline_m <= 0:
        return None
    return (focal_px * baseline_m) / disparity_px


def focal_from_known(
    disparity_px: float,
    known_z_m: float,
    baseline_m: float,
) -> float | None:
    if disparity_px <= 1.0 or known_z_m <= 0 or baseline_m <= 0:
        return None
    return (known_z_m * disparity_px) / baseline_m


def horizontal_disparity(
    cx_left: float,
    width_left: int,
    cx_right: float,
    width_right: int,
) -> float:
    """
    Disparity relative to each image centre.

    Handles different resolutions and is correct for parallel cameras
    even when widths differ (cx alone would be wrong across sizes).
    """
    return (cx_left - width_left / 2.0) - (cx_right - width_right / 2.0)


def load_saved_calib() -> dict | None:
    if not CALIB_PATH.exists():
        return None
    try:
        data = json.loads(CALIB_PATH.read_text())
        focal = float(data["focal_px"])
        if focal <= 0:
            return None
        return data
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_focal(
    focal_px: float,
    baseline_m: float,
    known_z_m: float,
    disparity_px: float,
    image_width: int,
) -> None:
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "calibrated_at_m": known_z_m,
        "disparity_px": disparity_px,
        "image_width": image_width,
        "note": "Simple parallel-stereo focal; redo C if cameras move or zoom changes",
    }
    CALIB_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved focal_px={focal_px:.1f} → {CALIB_PATH}")


def resolve_focal(
    cli_focal: float | None,
    baseline_m: float,
    image_width: int,
) -> float:
    if cli_focal is not None:
        if cli_focal <= 0:
            raise ValueError("--focal must be > 0")
        return cli_focal

    saved = load_saved_calib()
    if saved is not None:
        focal = float(saved["focal_px"])
        saved_w = int(saved.get("image_width") or image_width)
        if saved_w > 0 and saved_w != image_width:
            scale = image_width / saved_w
            focal *= scale
            print(
                f"Scaled saved focal for width {saved_w}→{image_width} "
                f"(focal_px={focal:.1f})"
            )
        saved_b = float(saved.get("baseline_m") or baseline_m)
        if abs(saved_b - baseline_m) > 0.01:
            print(
                f"WARNING: calib was for baseline {saved_b:.3f} m, "
                f"running with {baseline_m:.3f} m — press C to recalibrate."
            )
        print(f"Using saved focal_px={focal:.1f}")
        return focal

    focal = default_focal_px(image_width)
    print(
        f"Using rough default focal_px={focal:.1f}. "
        "Put the ball at the known distance and press C to calibrate."
    )
    return focal


def detect_ball(
    model: YOLO,
    frame,
) -> tuple[float, float, float, float, float, float, float] | None:
    """(cx, cy, x1, y1, x2, y2, conf) or None."""
    results = model.predict(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    best = None
    best_conf = -1.0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < best_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_conf = conf
        best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2, conf)
    return best


def paint(
    frame,
    title: str,
    det: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    cv2.putText(
        frame,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if det is None:
        cv2.putText(
            frame,
            "no ball",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return
    cx, cy, x1, y1, x2, y2, conf = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), -1)
    cv2.putText(
        frame,
        f"ball {conf:.2f}  cx={cx:.0f}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-camera football distance test")
    parser.add_argument("--left", type=int, default=0, help="Left camera index")
    parser.add_argument("--right", type=int, default=1, help="Right camera index")
    parser.add_argument(
        "--baseline",
        type=float,
        default=BASELINE_M,
        help="Camera separation in metres (default 1.0)",
    )
    parser.add_argument(
        "--known-distance",
        type=float,
        default=KNOWN_DISTANCE_M,
        help="True ball distance in metres when pressing C (default 5.0)",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Focal length in pixels (skips saved calib / default)",
    )
    parser.add_argument("--list", action="store_true", help="Probe camera indices 0–4 and exit")
    args = parser.parse_args()

    if args.list:
        for i in range(5):
            try:
                cap = open_camera(i)
                print(f"  OK index {i}")
                cap.release()
            except RuntimeError as exc:
                print(f"  -- index {i}: {exc}")
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

    print(
        f"Baseline={args.baseline:.3f} m | left={args.left} right={args.right}\n"
        f"Mount: LEFT=--left, RIGHT=--right, ~parallel, {args.baseline:.2f} m apart.\n"
        f"Calibrate: put ball at {args.known_distance:.2f} m, both must see it, press C.\n"
        "Q quit."
    )

    model = load_model()
    cap_l = None
    cap_r = None
    try:
        cap_l = open_camera(args.left)
        try:
            cap_r = open_camera(args.right)
        except RuntimeError:
            cap_l.release()
            raise

        ok_l, probe_l = read_frame(cap_l)
        ok_r, probe_r = read_frame(cap_r)
        if not ok_l or probe_l is None or not ok_r or probe_r is None:
            print("ERROR: couldn't read initial frames from both cameras.", file=sys.stderr)
            return 1

        if probe_l.shape[1] != probe_r.shape[1] or probe_l.shape[0] != probe_r.shape[0]:
            print(
                f"WARNING: different resolutions "
                f"L={probe_l.shape[1]}x{probe_l.shape[0]} "
                f"R={probe_r.shape[1]}x{probe_r.shape[0]} — "
                "using centre-relative disparity."
            )

        try:
            focal_px = resolve_focal(args.focal, args.baseline, int(probe_l.shape[1]))
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        depths: list[float] = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        while True:
            ok_l, frame_l = read_frame(cap_l)
            ok_r, frame_r = read_frame(cap_r)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                print("Frame grab failed; check USB / indices.")
                time.sleep(0.05)
                continue

            det_l = detect_ball(model, frame_l)
            det_r = detect_ball(model, frame_r)
            paint(frame_l, f"LEFT (cam {args.left})", det_l)
            paint(frame_r, f"RIGHT (cam {args.right})", det_r)

            status = "need ball in BOTH views"
            z_m: float | None = None

            if det_l is not None and det_r is not None:
                disparity = horizontal_disparity(
                    det_l[0],
                    frame_l.shape[1],
                    det_r[0],
                    frame_r.shape[1],
                )
                if disparity <= 1.0:
                    depths.clear()
                    status = (
                        f"bad disparity={disparity:.1f}px "
                        "(swap --left/--right or check aim)"
                    )
                else:
                    z_m = depth_from_disparity(disparity, focal_px, args.baseline)
                    if z_m is not None:
                        depths.append(z_m)
                        if len(depths) > SMOOTH_N:
                            depths.pop(0)
                        z_smooth = float(np.median(depths))
                        status = (
                            f"distance ≈ {z_smooth:.2f} m  "
                            f"(raw {z_m:.2f} m, disp {disparity:.1f}px, f={focal_px:.0f})"
                        )
            else:
                depths.clear()

            h = min(frame_l.shape[0], frame_r.shape[0])
            w = min(frame_l.shape[1], frame_r.shape[1])
            left = cv2.resize(frame_l, (w, h))
            right = cv2.resize(frame_r, (w, h))
            combo = np.hstack([left, right])
            cv2.rectangle(combo, (0, h - 40), (combo.shape[1], h), (0, 0, 0), -1)
            cv2.putText(
                combo,
                status,
                (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                if det_l is None or det_r is None:
                    print("Calibrate failed: ball not visible in both cameras.")
                    continue
                disp = horizontal_disparity(
                    det_l[0],
                    frame_l.shape[1],
                    det_r[0],
                    frame_r.shape[1],
                )
                if disp <= 1.0:
                    print(
                        f"Calibrate failed: disparity={disp:.1f}px. "
                        "Swap --left/--right if cameras are swapped."
                    )
                    continue
                new_f = focal_from_known(disp, args.known_distance, args.baseline)
                if new_f is None:
                    print("Calibrate failed: invalid numbers.")
                    continue
                focal_px = new_f
                depths.clear()
                save_focal(
                    focal_px,
                    args.baseline,
                    args.known_distance,
                    disp,
                    int(frame_l.shape[1]),
                )
                print(
                    f"Calibrated at {args.known_distance:.2f} m → "
                    f"focal_px={focal_px:.1f} (disparity={disp:.1f}px)"
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
