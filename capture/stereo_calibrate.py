"""
stereo_calibrate.py — Converging stereo calibration and real-time 3D ball position.

For cameras mounted on goalposts that point INWARD toward each other and tilted UPWARD.
This replaces the simple parallel-stereo test for the converging camera setup.

How it works
────────────
  Uses full 3D triangulation (OpenCV triangulatePoints) with known camera geometry:
    - Baseline : physical distance between cameras  (tape measure, e.g. 6.5 m)
    - H-angle  : how far each camera is rotated inward from straight-ahead (degrees)
    - V-angle  : how far each camera is tilted upward (degrees)
    - Focal    : lens focal length in pixels (loaded from old calib or estimated from FOV)

  The 3D world origin is the midpoint between the two cameras.
    X → positive toward the RIGHT camera (between posts: X in [-baseline/2, +baseline/2])
    Y → positive UPWARD
    Z → positive INTO THE FIELD (away from the goal)

Output shown in real time:
  Ball X  : lateral position relative to midpoint (0 = dead centre)
  Ball Y  : height above cameras
  Ball Z  : depth into field from goal line
  Verdict : BETWEEN POSTS / OUTSIDE LEFT / OUTSIDE RIGHT

Calibration (C key)
────────────────────
  Focal length is a LENS property — it doesn't change with angle or baseline.
  Your old focal (438 px at 480 px wide) scales automatically for resolution changes.
  You do NOT need to know the ball's size to calibrate.

  To calibrate or verify focal_px:
    1. Hold the ball at the MIDPOINT between the two cameras (equidistant from both).
    2. Measure how far it is above the camera bar with a tape measure.
       Pass that as --known-height (e.g. --known-height 1.5).
    3. Both cameras must see the ball. Press C.
       The script triangulates the ball's Y position and adjusts focal_px
       until the computed height matches your measurement. No ball size needed.

Saves to: exports/stereo/calib.json

Usage
─────
  python capture/stereo_calibrate.py --left 0 --right 1 --baseline 6.5
  python capture/stereo_calibrate.py --left 0 --right 1 --baseline 6.5 \\
      --h-angle 45 --v-angle 45 --known-distance 5.0

Keys: C = calibrate focal length | Q = quit and save current params
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

COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32
IMAGE_SIZE = 640
CONFIDENCE = 0.30
WINDOW = "Hawkeye Stereo Calibrate — C calibrate | Q save & quit"

SMOOTH_N = 6             # median smoothing window for 3D position

# Football diameter — used ONLY during C-key calibration. Cameras look up at the
# ball, so left sees the SW (bottom-left) face and right the SE (bottom-right);
# we nudge YOLO centres from those faces toward the true ball centre.
BALL_DIAMETER_M = 0.22
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0


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
        last_err = f"Camera {index} via {name}: no valid frames."
    raise RuntimeError(last_err)


def read_cam(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    return (True, frame) if (ok and frame is not None) else (False, None)


# ─── Projection matrix for a converging camera ───────────────────────────────

def make_proj_matrix(
    cam_pos_world: np.ndarray,
    h_deg: float,
    v_deg: float,
    focal_px: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Build a 3×4 OpenCV projection matrix P = K @ [R | t] for a camera
    positioned at cam_pos_world and pointing inward/upward.

    World frame:
      Origin = midpoint between cameras
      X = positive toward right camera
      Y = positive upward
      Z = positive into the field

    Camera convention (OpenCV):
      Camera X = right in image
      Camera Y = downward in image
      Camera Z = forward (depth)

    h_deg: horizontal aim (positive = look toward +X; ~90° = across goal)
    v_deg: vertical upward angle  (positive = look upward)

    Intrinsics use separate fx/fy from stereo_config mounted FOVs (sideways:
    ~47.3° horizontal, ~70° vertical). `focal_px` scales both so a C-key
    calibration still applies.
    """
    h = math.radians(h_deg)
    v = math.radians(v_deg)

    # Camera Z-axis (forward direction) in world coordinates
    cam_z = np.array([math.sin(h) * math.cos(v), math.sin(v), math.cos(h) * math.cos(v)])

    # Camera X-axis (right in image) in world: perpendicular to cam_z and world-up
    world_up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(world_up, cam_z)
    norm = np.linalg.norm(cam_x)
    if norm < 1e-6:
        cam_x = np.array([1.0, 0.0, 0.0])  # degenerate: camera pointing straight up
    else:
        cam_x /= norm

    # Camera Y-axis (downward in image) = cam_x × cam_z gives upward, negate for down
    cam_y_up = np.cross(cam_x, cam_z)
    cam_y_down = -cam_y_up / np.linalg.norm(cam_y_up)

    # Camera-to-world rotation (columns = camera axes in world)
    R_c2w = np.column_stack([cam_x, cam_y_down, cam_z])

    # World-to-camera rotation
    R_w2c = R_c2w.T

    # Translation: t = R_w2c @ (-cam_pos_world)
    t = R_w2c @ (-cam_pos_world)

    # Intrinsics: fx from image-HFOV, fy from image-VFOV (sideways mount)
    fx_spec, fy_spec = cfg.focal_axes(img_w, img_h)
    # Scale both axes by calibrated focal / spec fx (single C-key scalar)
    scale = (focal_px / fx_spec) if fx_spec > 1e-6 else 1.0
    fx = fx_spec * scale
    fy = fy_spec * scale
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    Rt = np.hstack([R_w2c, t.reshape(3, 1)])
    return K @ Rt


def triangulate_3d(
    P1: np.ndarray,
    P2: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
) -> np.ndarray | None:
    """
    Triangulate a 3D world point from two pixel observations.
    Returns (X, Y, Z) in world coordinates, or None if degenerate.
    """
    p1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    p2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)
    pts4d = cv2.triangulatePoints(P1, P2, p1, p2)
    w = float(np.asarray(pts4d[3]).reshape(-1)[0])
    if abs(w) < 1e-8:
        return None
    return (pts4d[:3] / w).flatten()


def project_world(P: np.ndarray, xyz: np.ndarray) -> tuple[float, float] | None:
    """Project a 3D world point with a 3×4 projection matrix → pixel (u, v)."""
    hom = P @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    if abs(hom[2]) < 1e-8:
        return None
    return float(hom[0] / hom[2]), float(hom[1] / hom[2])


def apparent_diameter_px(det: tuple) -> float:
    """Mean of bbox width and height in pixels."""
    _cx, _cy, x1, y1, x2, y2, _c = det
    return 0.5 * (abs(x2 - x1) + abs(y2 - y1))


def calib_centres_from_dets(
    det_l: tuple,
    det_r: tuple,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], str, np.ndarray | None]:
    """
    Calibration-only viewpoint correction.

    During C the cameras sit on the bar looking UP at the ball, so each view
    mainly sees the lower outer face of the ball:
      left cam  → south-west face (bottom + toward left cam)
      right cam → south-east face (bottom + toward right cam)

    YOLO box centres are assumed biased toward that near face. We estimate the
    true ball centre P, then for each camera nudge the image point from the
    projected near-face point toward the projected true centre.

    Returns (pt_l, pt_r, correction_summary, rough_pos).
    """
    # ~0.42 R is the centroid offset of a facing hemisphere from the sphere centre
    face_offset = BALL_RADIUS_M * (4.0 / (3.0 * math.pi))

    rough = triangulate_3d(P1, P2, (det_l[0], det_l[1]), (det_r[0], det_r[1]))
    if rough is None:
        return (float(det_l[0]), float(det_l[1])), (float(det_r[0]), float(det_r[1])), \
            "no rough 3D — skipped SW/SE correction", None

    P = rough.copy()
    # Prefer a physically plausible calib pose if triangulation is wild
    if float(P[1]) < 0.05:
        P[1] = max(float(P[1]), 0.3)

    def correct_one(
        det: tuple, cam: np.ndarray, Proj: np.ndarray, face_name: str,
    ) -> tuple[float, float, str]:
        to_ball = P - cam
        dist = float(np.linalg.norm(to_ball))
        if dist < 1e-3:
            return float(det[0]), float(det[1]), f"{face_name}: skip"
        direction = to_ball / dist  # camera → ball (near face faces -direction from centre)
        # Visual centre of the facing (SW/SE underside) hemisphere
        face_pt = P - face_offset * direction
        pix_centre = project_world(Proj, P)
        pix_face = project_world(Proj, face_pt)
        if pix_centre is None or pix_face is None:
            return float(det[0]), float(det[1]), f"{face_name}: project fail"
        du = pix_centre[0] - pix_face[0]
        dv = pix_centre[1] - pix_face[1]
        u = float(det[0]) + du
        v = float(det[1]) + dv
        return u, v, f"{face_name}: Δu={du:+.1f}px Δv={dv:+.1f}px"

    u_l, v_l, msg_l = correct_one(det_l, pos_l, P1, "L=SW")
    u_r, v_r, msg_r = correct_one(det_r, pos_r, P2, "R=SE")
    summary = f"{msg_l}; {msg_r}"
    return (u_l, v_l), (u_r, v_r), summary, rough


def between_posts_verdict(x: float, baseline: float, margin: float = 0.10) -> str:
    half = baseline / 2.0
    if -half - margin <= x <= half + margin:
        return f"BETWEEN POSTS  X={x:.2f} m"
    return f"OUTSIDE {'left' if x < 0 else 'right'}  X={x:.2f} m"


# ─── Focal length management — delegates to stereo_config ─────────────────────

def load_focal(img_w: int) -> tuple[float, str]:
    """Thin wrapper; priority: calib.json → simple_focal.json → spec formula."""
    f, src = cfg.get_focal_px(img_w)
    if "spec-derived" in src:
        src += "  (press C to calibrate)"
    return f, src


def save_calib(
    focal_px: float,
    baseline: float,
    h_angle: float,
    v_angle: float,
    img_w: int,
    img_h: int,
    known_dist: float | None = None,
) -> None:
    cfg.save_calib(
        focal_px=focal_px,
        baseline_m=baseline,
        h_angle_deg=h_angle,
        v_angle_deg=v_angle,
        image_width=img_w,
        image_height=img_h,
        calibrated_at_height_m=known_dist,
    )


# ─── Ball detection (COCO or custom model) ───────────────────────────────────

def load_model(model_str: str) -> tuple[YOLO, list[int] | None]:
    is_coco = (Path(model_str).name == COCO_MODEL)
    print(f"Loading {'COCO' if is_coco else 'custom'} model: {model_str}")
    return YOLO(model_str), ([SPORTS_BALL_CLASS] if is_coco else None)


def detect_both(
    model: YOLO,
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    conf: float,
    classes: list[int] | None,
) -> tuple[tuple | None, tuple | None]:
    kw: dict = dict(conf=conf, imgsz=IMAGE_SIZE, verbose=False)
    if classes:
        kw["classes"] = classes
    results = model.predict([frame_l, frame_r], **kw)
    return _parse_best(results[0]), _parse_best(results[1])


def _parse_best(result) -> tuple[float, float, float, float, float, float, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best, best_c = None, -1.0
    for box in result.boxes:
        c = float(box.conf[0])
        if c <= best_c:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_c = c
        best = ((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2, c)
    return best


# ─── Display helpers ─────────────────────────────────────────────────────────

def paint_det(frame: np.ndarray, det: tuple | None, color: tuple) -> None:
    if det is None:
        cv2.putText(frame, "no ball", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return
    cx, cy, x1, y1, x2, y2, conf = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 6, color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"ball {conf:.2f}", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def make_panel(frame_l: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
    h = min(frame_l.shape[0], frame_r.shape[0])
    def fit(f):
        if f.shape[0] == h:
            return f
        s = h / f.shape[0]
        return cv2.resize(f, (int(f.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    return np.hstack([fit(frame_l), fit(frame_r)])


def draw_status(combo: np.ndarray, lines: list[str], verdict: str = "") -> None:
    h, w = combo.shape[:2]
    bar_h = 28 * (len(lines) + (1 if verdict else 0)) + 10
    cv2.rectangle(combo, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(combo, line, (12, h - bar_h + 24 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if verdict:
        col = (0, 255, 80) if "BETWEEN" in verdict else (0, 80, 255)
        cv2.putText(combo, verdict, (12, h - bar_h + 24 + len(lines) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Converging stereo calibration")
    parser.add_argument("--left",  type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument(
        "--baseline", type=float, default=cfg.BASELINE_M,
        help=f"Physical distance between cameras in metres (from stereo_config). "
             f"Default {cfg.BASELINE_M}",
    )
    parser.add_argument(
        "--h-angle", type=float, default=cfg.H_ANGLE_DEG,
        help=f"Horizontal inward angle each camera makes with the bar (degrees). "
             f"Default {cfg.H_ANGLE_DEG}",
    )
    parser.add_argument(
        "--v-angle", type=float, default=cfg.V_ANGLE_DEG,
        help=f"Vertical upward angle each camera is tilted (degrees). "
             f"Default {cfg.V_ANGLE_DEG}",
    )
    parser.add_argument(
        "--known-height", type=float, default=0.60,
        help="Height of the ball ABOVE the camera-bar midpoint when pressing C (metres). "
             "Hold the ball on the centre line between the two cameras. Default 0.60 (60 cm)",
    )
    parser.add_argument(
        "--focal", type=float, default=None,
        help="Override focal length in pixels (skips auto-load)",
    )
    parser.add_argument(
        "--model", type=str, default=COCO_MODEL,
        help=f"Model file. Default {COCO_MODEL} (COCO, indoor). "
             "Use models/football_yolov8n.pt for outdoor sky.",
    )
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Probing cameras 0–5:")
        for i in range(6):
            try:
                cap = open_camera(i)
                print(f"  → {i} OK"); cap.release(); time.sleep(0.3)
            except RuntimeError as e:
                print(f"  → {i} FAIL: {e}")
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1
    if args.baseline <= 0:
        print("ERROR: --baseline must be > 0", file=sys.stderr)
        return 1
    if not (0 < args.h_angle < 90):
        print("ERROR: --h-angle must be in (0, 90)", file=sys.stderr)
        return 1
    if not (0 <= args.v_angle < 90):
        print("ERROR: --v-angle must be in [0, 90)", file=sys.stderr)
        return 1

    model_path = args.model
    if Path(model_path).name != COCO_MODEL:
        full = (PROJECT_ROOT / model_path) if not Path(model_path).is_absolute() else Path(model_path)
        if not full.exists():
            print(f"ERROR: model not found: {full}", file=sys.stderr)
            return 1
        model_path = str(full)

    model, classes = load_model(model_path)

    cap_l = cap_r = None
    try:
        print(f"\nOpening left camera ({args.left})...")
        cap_l = open_camera(args.left)
        time.sleep(0.5)
        print(f"Opening right camera ({args.right})...")
        cap_r = open_camera(args.right)

        ok_l, probe_l = read_cam(cap_l)
        ok_r, probe_r = read_cam(cap_r)
        if not ok_l or probe_l is None or not ok_r or probe_r is None:
            print("ERROR: could not read initial frames.", file=sys.stderr)
            return 1

        img_w = probe_l.shape[1]
        img_h = probe_l.shape[0]
        print(f"Left native: {probe_l.shape[1]}×{probe_l.shape[0]}")
        print(f"Right native: {probe_r.shape[1]}×{probe_r.shape[0]}")

        if args.focal is not None:
            focal_px = args.focal
            focal_src = "from --focal flag"
        else:
            focal_px, focal_src = load_focal(img_w)

        print(
            f"\nSetup:"
            f"\n  Baseline  : {args.baseline:.3f} m"
            f"\n  H-angle   : {args.h_angle:.1f}°  (each camera rotated inward)"
            f"\n  V-angle   : {args.v_angle:.1f}°  (each camera tilted upward)"
            f"\n  Focal     : {focal_px:.1f} px  ({focal_src})"
            f"\n  Ball diam : {BALL_DIAMETER_M*100:.0f} cm  (C-key only; SW/SE underside bias)"
            f"\n  Between-posts zone: X in [{-args.baseline/2:.2f}, {args.baseline/2:.2f}] m"
            "\n"
            "C = calibrate focal length:\n"
            "    Hold ball at the MIDPOINT between cameras at a measured height\n"
            "    (measure to the BALL CENTRE, above the camera-bar midpoint).\n"
            "    Cams look up → left sees SW (bottom-left) face, right sees SE (bottom-right).\n"
            f"    Uses {BALL_DIAMETER_M*100:.0f} cm diameter to correct that viewpoint bias.\n"
            f"    Current known-height = {args.known_height:.2f} m  (set with --known-height)"
            "\nQ = save & quit\n"
        )

        # Camera positions in world (origin = midpoint, X right, Y up, Z into field)
        pos_l = np.array([-args.baseline / 2, 0.0, 0.0])
        pos_r = np.array([ args.baseline / 2, 0.0, 0.0])

        # Build projection matrices — rebuilt each time focal_px changes
        def build_projs():
            P1 = make_proj_matrix(pos_l,  args.h_angle, args.v_angle, focal_px, img_w, img_h)
            P2 = make_proj_matrix(pos_r, -args.h_angle, args.v_angle, focal_px, img_w, img_h)
            return P1, P2

        P1, P2 = build_projs()

        positions_3d: list[np.ndarray] = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1200, 580)

        while True:
            ok_l, frame_l = read_cam(cap_l)
            ok_r, frame_r = read_cam(cap_r)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                time.sleep(0.02)
                continue

            det_l, det_r = detect_both(model, frame_l, frame_r, args.conf, classes)

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()
            cv2.putText(disp_l, f"LEFT  cam {args.left}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp_r, f"RIGHT cam {args.right}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 80, 200), 2, cv2.LINE_AA)
            paint_det(disp_l, det_l, (0, 255, 255))
            paint_det(disp_r, det_r, (255, 80, 200))

            status_lines = [f"f={focal_px:.0f} px | baseline={args.baseline:.2f} m | "
                            f"H={args.h_angle:.0f}° V={args.v_angle:.0f}°"]
            verdict = ""
            pos3d: np.ndarray | None = None

            if det_l is not None and det_r is not None:
                pos3d = triangulate_3d(P1, P2, (det_l[0], det_l[1]), (det_r[0], det_r[1]))
                if pos3d is not None:
                    positions_3d.append(pos3d)
                    if len(positions_3d) > SMOOTH_N:
                        positions_3d.pop(0)
                    smoothed = np.median(np.array(positions_3d), axis=0)
                    x, y, z = smoothed
                    status_lines.append(
                        f"3D pos:  X={x:+.2f} m  Y={y:+.2f} m  Z={z:+.2f} m"
                    )
                    verdict = between_posts_verdict(x, args.baseline)
            else:
                positions_3d.clear()
                status_lines.append("need ball in BOTH views for 3D position")

            combo = make_panel(disp_l, disp_r)
            draw_status(combo, status_lines, verdict)
            cv2.imshow(WINDOW, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                save_calib(focal_px, args.baseline, args.h_angle, args.v_angle,
                           img_w, img_h)
                break

            if key in (ord("c"), ord("C")):
                # ── Focal calibration ────────────────────────────────────────
                # Hold the ball at the MIDPOINT at --known-height (ball CENTRE).
                # Cameras look upward, so each mainly sees the lower outer face:
                #   left  → south-west (bottom + left)
                #   right → south-east (bottom + right)
                # Correct those biased YOLO centres using BALL_DIAMETER_M, then
                # scale focal so triangulated Y matches known height.
                # ─────────────────────────────────────────────────────────────
                if det_l is None or det_r is None:
                    print(
                        "C: need ball visible in BOTH cameras.\n"
                        "   Hold it at the midpoint between the cameras at measured height."
                    )
                    continue

                pt_l, pt_r, corr_summary, _rough = calib_centres_from_dets(
                    det_l, det_r, pos_l, pos_r, P1, P2,
                )
                trial_pos = triangulate_3d(P1, P2, pt_l, pt_r)
                if trial_pos is None:
                    print("C: triangulation failed — cameras may be too parallel for this setup.")
                    continue

                computed_y = float(trial_pos[1])
                known_h = args.known_height
                diam_l = apparent_diameter_px(det_l)
                diam_r = apparent_diameter_px(det_r)

                if computed_y <= 0.05:
                    print(
                        f"C: REJECTED — computed height is {computed_y:.3f} m "
                        f"(need clearly positive Y ≈ {known_h:.2f} m).\n"
                        "   Ball may not be above the bar midpoint, or left/right "
                        "cameras may be swapped, or --v-angle is wrong.\n"
                        "   Do not trust this pose — fix placement and try C again."
                    )
                    continue

                scale = known_h / computed_y
                new_focal = focal_px * scale
                if not (80.0 <= new_focal <= 4000.0) or scale < 0.05 or scale > 20.0:
                    print(
                        f"C: REJECTED — absurd focal update "
                        f"(computed_Y={computed_y:.3f} m, scale={scale:.3f}, "
                        f"focal {focal_px:.1f} → {new_focal:.1f}).\n"
                        "   Check ball is centred between cams at the known height."
                    )
                    continue

                focal_px = new_focal
                P1, P2 = build_projs()
                positions_3d.clear()
                save_calib(focal_px, args.baseline, args.h_angle, args.v_angle,
                           img_w, img_h, known_dist=known_h)
                print(
                    f"Calibrated: computed_Y={computed_y:.3f} m → known={known_h:.3f} m "
                    f"(scale={scale:.3f}) → focal_px={focal_px:.1f}\n"
                    f"  SW/SE underside correction ({BALL_DIAMETER_M*100:.0f} cm ball): {corr_summary}\n"
                    f"  bbox diam L={diam_l:.0f}px R={diam_r:.0f}px\n"
                    f"  Ball 3D position at calibration: "
                    f"X={trial_pos[0]*scale:.3f} m  Y={known_h:.3f} m  Z={trial_pos[2]*scale:.3f} m"
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
