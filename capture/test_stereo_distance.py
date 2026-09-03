"""
Simple dual-camera distance test (still football, indoor).

Assumes:
  - Two cameras on a rigid baseline (default 1.0 m apart)
  - Roughly parallel, both looking at the ball
  - Left camera = --left, right camera = --right

Uses the same capture settings as record_video.py:
  1280x720, sky-band SoftwareAutoExposure (brightness / exposure / gain).

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

Indoor / new environment (ball not detected):
  - Lower confidence: --conf 0.25 (model was trained outdoors on sky)
  - Fix sideways sensor: --rotate 90 or --rotate -90 (try both)
  - Verify one camera first: python capture/test_detect.py
  - Brighter ball, plain background, move ball to centre of frame
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

from auto_exposure import SoftwareAutoExposure, read_frame, try_set


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
WARMUP_FRAMES = 10
READ_RETRIES = 30
WINDOW_STEREO = "Hawkeye Stereo — C calibrate | Q quit"
DISPLAY_SCALE = 0.5  # each cam scaled before side-by-side panel


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
    # Windows — MSMF is better for Arducam exposure controls; DSHOW as fallback.
    return [
        ("MSMF", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
        ("default", cv2.CAP_ANY),
    ]


def rotate_frame(frame, degrees: int):
    """Rotate captured frame so the wide FOV is horizontal (cam mounted sideways)."""
    if degrees == 0:
        return frame
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees in (-90, 270):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"Unsupported --rotate {degrees}; use 0, 90, -90, or 180")


def capture_resolution(swap_res: bool) -> tuple[int, int]:
    """Request WxH from the driver; --swap-res asks for portrait before rotating."""
    width, height = FRAME_WIDTH, FRAME_HEIGHT
    if swap_res:
        width, height = height, width
    return width, height


def open_camera(index: int, swap_res: bool = False) -> cv2.VideoCapture:
    """Open USB camera with the same backends / resolution as record_video.py."""
    last_error = f"Couldn't open camera {index}."
    req_w, req_h = capture_resolution(swap_res)

    for name, backend in camera_backends():
        print(f"Trying camera {index} via {name}...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if req_w is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        if req_h is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = read_frame(cap, retries=max(WARMUP_FRAMES, 1))
        if ok and frame is not None:
            print(
                f"Camera {index} opened via {name} "
                f"({frame.shape[1]}x{frame.shape[0]})"
            )
            return cap

        cap.release()
        last_error = (
            f"Camera {index} opened via {name} but no frames came through. "
            "Close other camera apps, unplug/replug, then try again."
        )

    raise RuntimeError(last_error)


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
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float, float, float, float, float] | None:
    """(cx, cy, x1, y1, x2, y2, conf) or None."""
    results = model.predict(
        frame,
        conf=min(confidence, 0.15),
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
    if best is None or best_conf < confidence:
        return None
    return best


def paint(
    frame,
    title: str,
    det: tuple[float, float, float, float, float, float, float] | None,
    status: str = "",
    exposure_text: str = "",
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
    else:
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

    if exposure_text:
        cv2.putText(
            frame,
            exposure_text,
            (12, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    if status:
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status,
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def create_resizable_window(name: str, width: int, height: int) -> None:
    """OpenCV window the user can freely resize; feed scales to match."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    except cv2.error:
        pass
    cv2.resizeWindow(name, width, height)


def imshow_fitted(window_name: str, frame) -> None:
    """Show frame stretched to the current window size so resizing works."""
    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        win_w, win_h = 0, 0

    if win_w > 1 and win_h > 1 and (win_w != frame.shape[1] or win_h != frame.shape[0]):
        display = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_AREA)
    else:
        display = frame
    cv2.imshow(window_name, display)


def build_stereo_panel(frame_l, frame_r, scale: float):
    """Scale each feed and stack left | right in one image."""
    if scale <= 0 or scale > 1.0:
        raise ValueError("--display-scale must be in (0, 1]")
    if scale != 1.0:
        frame_l = cv2.resize(
            frame_l,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        frame_r = cv2.resize(
            frame_r,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return np.hstack([frame_l, frame_r])


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
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
        help=f"Detection confidence threshold (default {CONFIDENCE}; try 0.25 indoors)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=90,
        choices=[0, 90, -90, 180],
        help="Rotate frames so wide FOV is horizontal (default 90; use -90 if upside-down)",
    )
    parser.add_argument(
        "--swap-res",
        action="store_true",
        help="Request portrait resolution from camera before --rotate (try if still sideways)",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=DISPLAY_SCALE,
        help=f"Scale each camera before side-by-side panel (default {DISPLAY_SCALE})",
    )
    args = parser.parse_args()

    if args.list:
        for i in range(5):
            try:
                cap = open_camera(i, swap_res=False)
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
    if not 0.0 < args.conf <= 1.0:
        print("ERROR: --conf must be in (0, 1]", file=sys.stderr)
        return 1
    if args.display_scale <= 0 or args.display_scale > 1.0:
        print("ERROR: --display-scale must be in (0, 1]", file=sys.stderr)
        return 1

    print(
        f"Baseline={args.baseline:.3f} m | left={args.left} right={args.right}\n"
        f"Mount: LEFT=--left, RIGHT=--right, ~parallel, {args.baseline:.2f} m apart.\n"
        f"Calibrate: put ball at {args.known_distance:.2f} m, both must see it, press C.\n"
        f"Rotate={args.rotate}° | conf={args.conf:.2f} | display-scale={args.display_scale:.2f}\n"
        "Q quit."
    )

    model = load_model()
    cap_l = None
    cap_r = None
    try:
        cap_l = open_camera(args.left, swap_res=args.swap_res)
        try:
            cap_r = open_camera(args.right, swap_res=args.swap_res)
        except RuntimeError:
            cap_l.release()
            raise

        ok_l, probe_l = read_frame(cap_l)
        ok_r, probe_r = read_frame(cap_r)
        if not ok_l or probe_l is None or not ok_r or probe_r is None:
            print("ERROR: couldn't read initial frames from both cameras.", file=sys.stderr)
            return 1

        probe_l = rotate_frame(probe_l, args.rotate)
        probe_r = rotate_frame(probe_r, args.rotate)

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

        print("Setting up sky auto-exposure on both cameras (same as record_video)...")
        auto_l = SoftwareAutoExposure(cap_l)
        auto_r = SoftwareAutoExposure(cap_r)
        auto_l.settle()
        auto_r.settle()

        depths: list[float] = []
        panel_w = max(1, int(probe_l.shape[1] * args.display_scale * 2))
        panel_h = max(1, int(probe_l.shape[0] * args.display_scale))
        create_resizable_window(WINDOW_STEREO, panel_w, panel_h)

        while True:
            ok_l, raw_l = read_frame(cap_l, retries=READ_RETRIES)
            ok_r, raw_r = read_frame(cap_r, retries=READ_RETRIES)
            if not ok_l or raw_l is None or not ok_r or raw_r is None:
                print("Frame grab failed; check USB / indices.")
                time.sleep(0.05)
                continue

            raw_l = rotate_frame(raw_l, args.rotate)
            raw_r = rotate_frame(raw_r, args.rotate)
            frame_l = auto_l.process(raw_l)
            frame_r = auto_r.process(raw_r)

            det_l = detect_ball(model, frame_l, confidence=args.conf)
            det_r = detect_ball(model, frame_r, confidence=args.conf)

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

            paint(
                frame_l,
                f"LEFT (cam {args.left}) {frame_l.shape[1]}x{frame_l.shape[0]}",
                det_l,
                status=status,
                exposure_text=auto_l.status_text(),
            )
            paint(
                frame_r,
                f"RIGHT (cam {args.right}) {frame_r.shape[1]}x{frame_r.shape[0]}",
                det_r,
                status=status,
                exposure_text=auto_r.status_text(),
            )
            panel = build_stereo_panel(frame_l, frame_r, args.display_scale)
            imshow_fitted(WINDOW_STEREO, panel)

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
