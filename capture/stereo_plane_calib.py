"""
stereo_plane_calib.py — Draw / verify the goal plane (test only).

Does NOT change stereo_record or stereo_calibrate. Use this to check that
the goal plane (between the posts, extended upward) projects correctly
into both camera views with your current stereo_config angles.

Assumptions
───────────
  Cameras sit at the post–crossbar corners.
  Goal plane = plane containing both cameras and world-up  (Z = 0).
  Left post  at X = −baseline/2,  right post at X = +baseline/2.
  Crossbar height ≈ camera height (Y = 0). Plane extends UP for high balls.

  Config aim (edit stereo_config.py):
    H_ANGLE ≈ 90°  → each camera looks across the goal toward the far post
    V_ANGLE ≈ 55°  → tilted upward

Controls
────────
  Click LEFT panel  = mark a point on the LEFT camera image
  Click RIGHT panel = mark a point on the RIGHT camera image

  F = next clicks are FAR post (default)
  N = next clicks are NEAR post
  C = clear all click marks

  [ / ] = decrease / increase H_ANGLE by 1°
  - / = = decrease / increase V_ANGLE by 1°
  , / . = decrease / increase plane height extension (metres)
  R     = reset angles to stereo_config defaults
  B     = toggle ball detection (COCO sports-ball) + plane-side readout
  D     = toggle landmark distance labels on the plane
  S     = save plane overlay params → exports/stereo/plane.json
  L     = load plane.json if present
  Q/ESC = quit

Usage
─────
  python capture/stereo_plane_calib.py --left 0 --right 1
  python capture/stereo_plane_calib.py --list
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
from stereo_calibrate import make_proj_matrix, project_world, triangulate_3d


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANE_PATH = PROJECT_ROOT / "exports" / "stereo" / "plane.json"

COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32
IMAGE_SIZE = 640
CONFIDENCE = 0.30
WINDOW = "Hawkeye Plane Calib — click posts | [ ] H | - = V | D labels | B ball | S save | Q"

# Drawing
COL_GRID = (80, 200, 80)
COL_POST = (0, 255, 255)
COL_BAR = (0, 180, 255)
COL_EXT = (180, 255, 100)
COL_CLICK_FAR = (0, 255, 0)
COL_CLICK_NEAR = (255, 128, 0)
COL_BALL = (0, 255, 255)
COL_STATUS = (220, 220, 220)
COL_DEBUG = (180, 255, 255)
COL_DIST = (255, 255, 100)


def dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def elevation_from_horizontal(cam_pos: np.ndarray, xyz: np.ndarray) -> float:
    """Elevation angle of cam→point above the horizontal plane (degrees)."""
    d = xyz - cam_pos
    horiz = math.hypot(float(d[0]), float(d[2]))
    if horiz < 1e-9:
        return 90.0 if d[1] > 0 else (-90.0 if d[1] < 0 else 0.0)
    return math.degrees(math.atan2(float(d[1]), horiz))


def put_label(img: np.ndarray, text: str, org: tuple[int, int],
              color=COL_DIST, scale=0.42, thick=1) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def label_world_point(
    img: np.ndarray,
    P: np.ndarray,
    xyz: np.ndarray,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    name: str,
    which_cam: str,
) -> None:
    """
    Project a world point and annotate distance to BOTH cameras plus elevation
    from *this* camera (the feed we're drawing on).
    """
    uv = project_safe(P, xyz)
    if uv is None:
        return
    u, v = uv
    if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
        return
    d_l = dist3(xyz, pos_l)
    d_r = dist3(xyz, pos_r)
    cam = pos_l if which_cam == "L" else pos_r
    elev = elevation_from_horizontal(cam, xyz)
    cv2.circle(img, (u, v), 5, COL_DIST, -1, cv2.LINE_AA)
    # Two lines so it stays readable
    put_label(img, f"{name}", (u + 8, v - 18), COL_POST, 0.45, 1)
    put_label(
        img,
        f"dL={d_l:.2f}m dR={d_r:.2f}m  elev={elev:.0f}deg",
        (u + 8, v + 4),
        COL_DIST,
        0.38,
        1,
    )


# ─── Camera helpers (same pattern as other stereo scripts) ───────────────────

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


# ─── Goal plane geometry ─────────────────────────────────────────────────────

def goal_corners(
    baseline_m: float,
    extend_up_m: float,
    below_m: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    World points on the goal plane (Z = 0).
    Y = 0 at camera / crossbar height; positive Y is upward (above the bar).
    """
    half = baseline_m / 2.0
    return {
        "left_base": np.array([-half, -below_m, 0.0]),
        "right_base": np.array([half, -below_m, 0.0]),
        "left_top": np.array([-half, extend_up_m, 0.0]),
        "right_top": np.array([half, extend_up_m, 0.0]),
        "bar_left": np.array([-half, 0.0, 0.0]),
        "bar_right": np.array([half, 0.0, 0.0]),
    }


def project_safe(P: np.ndarray, xyz: np.ndarray) -> tuple[int, int] | None:
    uv = project_world(P, xyz)
    if uv is None:
        return None
    u, v = uv
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    # Reject points behind the camera (negative depth in camera frame ≈ hom[2] sign
    # already handled in project_world via division; still clamp absurd values)
    if abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return int(round(u)), int(round(v))


def draw_line_world(
    img: np.ndarray,
    P: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    n_seg: int = 24,
) -> None:
    """Draw a 3D segment by sampling so perspective looks correct."""
    pts: list[tuple[int, int]] = []
    for i in range(n_seg + 1):
        t = i / n_seg
        xyz = a * (1.0 - t) + b * t
        uv = project_safe(P, xyz)
        if uv is not None:
            pts.append(uv)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


def draw_goal_plane(
    img: np.ndarray,
    P: np.ndarray,
    baseline_m: float,
    extend_up_m: float,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    which_cam: str,
    grid_nx: int = 6,
    grid_ny: int = 8,
    show_point_labels: bool = True,
) -> None:
    """Project uprights, crossbar, upward grid, and per-point distance labels."""
    half = baseline_m / 2.0
    corners = goal_corners(baseline_m, extend_up_m)

    # Uprights (extended above bar)
    draw_line_world(img, P, corners["bar_left"], corners["left_top"], COL_EXT, 2)
    draw_line_world(img, P, corners["bar_right"], corners["right_top"], COL_EXT, 2)
    draw_line_world(img, P, corners["left_base"], corners["bar_left"], COL_POST, 3)
    draw_line_world(img, P, corners["right_base"], corners["bar_right"], COL_POST, 3)
    draw_line_world(img, P, corners["bar_left"], corners["bar_right"], COL_BAR, 3)
    draw_line_world(img, P, corners["left_top"], corners["right_top"], COL_GRID, 1)

    for i in range(grid_nx + 1):
        x = -half + (baseline_m * i / grid_nx)
        a = np.array([x, 0.0, 0.0])
        b = np.array([x, extend_up_m, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=16)

    for j in range(1, grid_ny + 1):
        y = extend_up_m * j / grid_ny
        a = np.array([-half, y, 0.0])
        b = np.array([half, y, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=16)

    if not show_point_labels:
        return

    # Landmark heights on each post: bar, lowest-ray hit, mid, top of extension
    ray_deg, h_low, _slant = cfg.lowest_visible_on_far_post(baseline_m)
    h_low = max(0.05, h_low)  # avoid sitting exactly on the bar label
    landmarks: list[tuple[str, np.ndarray]] = [
        ("L-bar", np.array([-half, 0.0, 0.0])),
        ("R-bar", np.array([half, 0.0, 0.0])),
        ("mid-bar", np.array([0.0, 0.0, 0.0])),
        (f"L@{h_low:.2f}m", np.array([-half, h_low, 0.0])),
        (f"R@{h_low:.2f}m", np.array([half, h_low, 0.0])),
        ("L-mid", np.array([-half, extend_up_m * 0.5, 0.0])),
        ("R-mid", np.array([half, extend_up_m * 0.5, 0.0])),
        ("L-top", np.array([-half, extend_up_m, 0.0])),
        ("R-top", np.array([half, extend_up_m, 0.0])),
        ("mid-top", np.array([0.0, extend_up_m, 0.0])),
    ]
    # From each camera, the FAR post is the important one — bold those first
    far_name = "R" if which_cam == "L" else "L"
    for name, xyz in landmarks:
        if name.startswith(far_name) or name.startswith("mid") or "@" in name:
            label_world_point(img, P, xyz, pos_l, pos_r, name, which_cam)


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)


def make_panel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    return np.hstack([fit_height(left, h), fit_height(right, h)])


# ─── Click state ─────────────────────────────────────────────────────────────

class ClickState:
    def __init__(self) -> None:
        self.mode = "far"  # "far" | "near"
        self.far_l: list[tuple[int, int]] = []
        self.far_r: list[tuple[int, int]] = []
        self.near_l: list[tuple[int, int]] = []
        self.near_r: list[tuple[int, int]] = []

    def add(self, side: str, xy: tuple[int, int]) -> None:
        bucket = {
            ("far", "L"): self.far_l,
            ("far", "R"): self.far_r,
            ("near", "L"): self.near_l,
            ("near", "R"): self.near_r,
        }[(self.mode, side)]
        bucket.append(xy)
        print(f"  Marked {self.mode.upper()} post on {side}: {xy}  (n={len(bucket)})")

    def clear(self) -> None:
        self.far_l.clear()
        self.far_r.clear()
        self.near_l.clear()
        self.near_r.clear()
        print("  Cleared all click marks")

    def draw_on(
        self,
        img_l: np.ndarray,
        img_r: np.ndarray,
        P1: np.ndarray | None = None,
        P2: np.ndarray | None = None,
        pos_l: np.ndarray | None = None,
        pos_r: np.ndarray | None = None,
    ) -> list[str]:
        """Draw clicks; if paired L/R exist, triangulate and annotate distances."""
        debug_lines: list[str] = []

        def _draw(img, pts, color, label):
            for i, (x, y) in enumerate(pts):
                cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
                put_label(img, f"{label}{i+1}", (x + 8, y - 8), color, 0.5, 1)
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

        _draw(img_l, self.far_l, COL_CLICK_FAR, "F")
        _draw(img_r, self.far_r, COL_CLICK_FAR, "F")
        _draw(img_l, self.near_l, COL_CLICK_NEAR, "N")
        _draw(img_r, self.near_r, COL_CLICK_NEAR, "N")

        if P1 is None or P2 is None or pos_l is None or pos_r is None:
            return debug_lines

        def _pair(name: str, pts_l: list, pts_r: list, color):
            n = min(len(pts_l), len(pts_r))
            for i in range(n):
                xyz = triangulate_3d(P1, P2, pts_l[i], pts_r[i])
                if xyz is None:
                    debug_lines.append(f"{name}{i+1}: triangulate failed")
                    continue
                d_l = dist3(xyz, pos_l)
                d_r = dist3(xyz, pos_r)
                elev_l = elevation_from_horizontal(pos_l, xyz)
                elev_r = elevation_from_horizontal(pos_r, xyz)
                x, y, z = map(float, xyz)
                msg = (
                    f"{name}{i+1}: XYZ=({x:+.2f},{y:+.2f},{z:+.2f})  "
                    f"dL={d_l:.2f}m dR={d_r:.2f}m  elevL={elev_l:.0f}° elevR={elev_r:.0f}°"
                )
                debug_lines.append(msg)
                for img, P, pt in ((img_l, P1, pts_l[i]), (img_r, P2, pts_r[i])):
                    put_label(
                        img,
                        f"dL={d_l:.2f} dR={d_r:.2f} Y={y:.2f}",
                        (pt[0] + 8, pt[1] + 16),
                        color,
                        0.4,
                        1,
                    )

        _pair("Far", self.far_l, self.far_r, COL_CLICK_FAR)
        _pair("Near", self.near_l, self.near_r, COL_CLICK_NEAR)
        return debug_lines

    def to_dict(self) -> dict:
        return {
            "far_left": self.far_l,
            "far_right": self.far_r,
            "near_left": self.near_l,
            "near_right": self.near_r,
        }

    def load_dict(self, data: dict) -> None:
        self.far_l = [tuple(p) for p in data.get("far_left", [])]
        self.far_r = [tuple(p) for p in data.get("far_right", [])]
        self.near_l = [tuple(p) for p in data.get("near_left", [])]
        self.near_r = [tuple(p) for p in data.get("near_right", [])]


# ─── Ball (optional) ─────────────────────────────────────────────────────────

def detect_both(model: YOLO, frame_l, frame_r, conf: float):
    results = model.predict(
        [frame_l, frame_r], conf=conf, classes=[SPORTS_BALL_CLASS],
        imgsz=IMAGE_SIZE, verbose=False,
    )

    def best(res):
        if res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes
        i = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[i].cpu().numpy()
        c = float(boxes.conf[i])
        cx = 0.5 * (xyxy[0] + xyxy[2])
        cy = 0.5 * (xyxy[1] + xyxy[3])
        return cx, cy, c, xyxy

    return best(results[0]), best(results[1])


def plane_side_label(z: float, eps: float = 0.05) -> str:
    if z > eps:
        return f"INTO FIELD  Z={z:+.2f} m"
    if z < -eps:
        return f"BEHIND GOAL  Z={z:+.2f} m"
    return f"ON PLANE  Z={z:+.2f} m"


# ─── Save / load ─────────────────────────────────────────────────────────────

def save_plane(
    baseline_m: float,
    h_angle: float,
    v_angle: float,
    extend_up_m: float,
    focal_px: float,
    img_w: int,
    img_h: int,
    clicks: ClickState,
) -> None:
    PLANE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_m": baseline_m,
        "h_angle_deg": h_angle,
        "v_angle_deg": v_angle,
        "extend_up_m": extend_up_m,
        "focal_px": focal_px,
        "image_width": img_w,
        "image_height": img_h,
        "plane": {
            "origin": "midpoint between cameras",
            "equation": "Z = 0 (goal plane / posts / crossbar)",
            "left_post_x": -baseline_m / 2.0,
            "right_post_x": baseline_m / 2.0,
            "crossbar_y": 0.0,
            "note": "Y > 0 is above the bar; Z > 0 is into the field",
        },
        "clicks": clicks.to_dict(),
        "camera": "B0332 + LN013 NOIR",
    }
    PLANE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {PLANE_PATH}")
    print(json.dumps(payload, indent=2))


def load_plane() -> dict | None:
    if not PLANE_PATH.exists():
        return None
    try:
        return json.loads(PLANE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Goal-plane overlay / click calibration test")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--baseline", type=float, default=cfg.BASELINE_M)
    parser.add_argument("--h-angle", type=float, default=cfg.H_ANGLE_DEG)
    parser.add_argument("--v-angle", type=float, default=cfg.V_ANGLE_DEG)
    parser.add_argument("--extend", type=float, default=4.0,
                        help="How far above the bar to draw the plane (metres)")
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Probing cameras 0–5:")
        for i in range(6):
            try:
                cap = open_camera(i)
                print(f"  → {i} OK")
                cap.release()
                time.sleep(0.3)
            except RuntimeError as e:
                print(f"  → {i} FAIL: {e}")
        return 0

    if args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1

    h_angle = float(args.h_angle)
    v_angle = float(args.v_angle)
    extend_up = float(args.extend)
    baseline = float(args.baseline)
    clicks = ClickState()
    ball_on = False
    show_labels = True
    model: YOLO | None = None
    panel_split_x = 0
    frame_i = 0
    fps_ema = 0.0
    last_t = time.time()

    mouse = {"x": 0, "y": 0, "clicked": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["x"] = x
            mouse["y"] = y
            mouse["clicked"] = True

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

        img_w, img_h = probe_l.shape[1], probe_l.shape[0]
        if args.focal is not None:
            focal_px, focal_src = float(args.focal), "CLI"
        else:
            focal_px, focal_src = cfg.get_focal_px(img_w)

        cfg.print_summary(focal_px, focal_src)
        print(
            f"Plane test defaults: H={h_angle:.1f}°  V={v_angle:.1f}°  "
            f"extend={extend_up:.1f} m above bar\n"
            "Click far post(s) in each view. Use [ ] and - = to align the green grid "
            "with the real posts.\n"
        )

        pos_l = np.array([-baseline / 2.0, 0.0, 0.0])
        pos_r = np.array([baseline / 2.0, 0.0, 0.0])

        def build_projs():
            # Left looks across toward +X (far/right post); right toward −X
            P1 = make_proj_matrix(pos_l, +h_angle, v_angle, focal_px, img_w, img_h)
            P2 = make_proj_matrix(pos_r, -h_angle, v_angle, focal_px, img_w, img_h)
            return P1, P2

        P1, P2 = build_projs()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 600)
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            ok_l, frame_l = read_cam(cap_l)
            ok_r, frame_r = read_cam(cap_r)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                time.sleep(0.02)
                continue

            # Rebuild projection if angles changed
            P1, P2 = build_projs()
            fx_now, fy_now = cfg.focal_axes(img_w, img_h)
            scale = (focal_px / fx_now) if fx_now > 1e-6 else 1.0
            fx_use, fy_use = fx_now * scale, fy_now * scale
            ray_deg, h_low, slant = cfg.lowest_visible_on_far_post(baseline)
            # Recompute lowest ray from live V angle (config helper uses config V)
            live_ray = v_angle - (cfg.VFOV_DEG / 2.0)
            live_h = baseline * math.tan(math.radians(live_ray)) if live_ray > -89 else 0.0
            live_slant = baseline / math.cos(math.radians(live_ray)) if abs(live_ray) < 89 else float("inf")

            now = time.time()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps_ema = (1.0 / dt) if frame_i == 0 else (0.9 * fps_ema + 0.1 / dt)
            frame_i += 1

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()
            draw_goal_plane(
                disp_l, P1, baseline, extend_up, pos_l, pos_r, "L",
                show_point_labels=show_labels,
            )
            draw_goal_plane(
                disp_r, P2, baseline, extend_up, pos_l, pos_r, "R",
                show_point_labels=show_labels,
            )
            click_dbg = clicks.draw_on(disp_l, disp_r, P1, P2, pos_l, pos_r)

            # Per-camera HUD (top-left under title)
            for disp, tag, pos_self, pos_other in (
                (disp_l, "LEFT", pos_l, pos_r),
                (disp_r, "RIGHT", pos_r, pos_l),
            ):
                far = pos_other  # far post ≈ other camera mount
                d_far_bar = dist3(far, pos_self)
                elev_far_bar = elevation_from_horizontal(pos_self, far)
                far_low = far + np.array([0.0, max(live_h, 0.0), 0.0])
                d_far_low = dist3(far_low, pos_self)
                elev_far_low = elevation_from_horizontal(pos_self, far_low)
                hud = [
                    f"{tag}  cam {args.left if tag == 'LEFT' else args.right}  "
                    f"{img_w}x{img_h}",
                    f"far-bar: {d_far_bar:.2f}m  elev {elev_far_bar:.0f}deg",
                    f"far@low: {d_far_low:.2f}m  elev {elev_far_low:.0f}deg  "
                    f"(Y={live_h:.2f}m)",
                    f"to other cam: {dist3(pos_self, pos_other):.2f}m",
                ]
                for li, line in enumerate(hud):
                    put_label(disp, line, (12, 28 + li * 18), COL_DEBUG, 0.48, 1)

            ball_line = ""
            if ball_on:
                if model is None:
                    print("Loading COCO yolov8n for ball...")
                    model = YOLO(COCO_MODEL)
                det_l, det_r = detect_both(model, frame_l, frame_r, CONFIDENCE)
                if det_l is not None:
                    x1, y1, x2, y2 = map(int, det_l[3])
                    cv2.rectangle(disp_l, (x1, y1), (x2, y2), COL_BALL, 2)
                    cv2.circle(disp_l, (int(det_l[0]), int(det_l[1])), 5, COL_BALL, -1)
                if det_r is not None:
                    x1, y1, x2, y2 = map(int, det_r[3])
                    cv2.rectangle(disp_r, (x1, y1), (x2, y2), COL_BALL, 2)
                    cv2.circle(disp_r, (int(det_r[0]), int(det_r[1])), 5, COL_BALL, -1)
                if det_l is not None and det_r is not None:
                    pos3d = triangulate_3d(
                        P1, P2, (det_l[0], det_l[1]), (det_r[0], det_r[1]),
                    )
                    if pos3d is not None:
                        x, y, z = map(float, pos3d)
                        d_l = dist3(pos3d, pos_l)
                        d_r = dist3(pos3d, pos_r)
                        side = plane_side_label(z)
                        between = abs(x) <= baseline / 2.0 + 0.05
                        above = y > -0.05
                        ball_line = (
                            f"Ball XYZ=({x:+.2f},{y:+.2f},{z:+.2f})  "
                            f"dL={d_l:.2f}m dR={d_r:.2f}m  |  {side}  |  "
                            f"{'BETWEEN' if between else 'OUTSIDE'}  "
                            f"{'ABOVE' if above else 'BELOW'}"
                        )
                        put_label(
                            disp_l,
                            f"ball dL={d_l:.2f} dR={d_r:.2f}",
                            (int(det_l[0]) + 8, int(det_l[1]) + 20),
                            COL_BALL, 0.45, 1,
                        )
                        put_label(
                            disp_r,
                            f"ball dL={d_l:.2f} dR={d_r:.2f}",
                            (int(det_r[0]) + 8, int(det_r[1]) + 20),
                            COL_BALL, 0.45, 1,
                        )

            fl = fit_height(disp_l, max(disp_l.shape[0], disp_r.shape[0]))
            fr = fit_height(disp_r, fl.shape[0])
            combo = np.hstack([fl, fr])
            panel_split_x = fl.shape[1]

            # Status / debug bar
            lines = [
                f"H={h_angle:.1f}° V={v_angle:.1f}°  base={baseline:.2f}m  "
                f"extend={extend_up:.1f}m  fx/fy={fx_use:.0f}/{fy_use:.0f}  "
                f"FOV H/V={cfg.HFOV_DEG:.1f}/{cfg.VFOV_DEG:.1f}°  "
                f"{fps_ema:.0f} fps  labels={'ON' if show_labels else 'OFF'}",
                f"Lowest ray={live_ray:.1f}°  far-post lowest Y={live_h:.2f}m  "
                f"slant={live_slant:.2f}m  (expect ~{slant:.2f}m @ cfg)  "
                f"mode={clicks.mode.upper()}",
                "F/N mode | C clear | [ ] H | - = V | , . extend | D labels | "
                "B ball | S save | L load | R reset | Q quit",
            ]
            if ball_line:
                lines.append(ball_line)
            lines.extend(click_dbg[:4])  # up to 4 triangulated click summaries
            bar_h = 20 * len(lines) + 10
            overlay = combo.copy()
            cv2.rectangle(overlay, (0, combo.shape[0] - bar_h), (combo.shape[1], combo.shape[0]),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, combo, 0.45, 0, combo)
            for i, line in enumerate(lines):
                cv2.putText(
                    combo, line, (10, combo.shape[0] - bar_h + 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_STATUS, 1, cv2.LINE_AA,
                )

            cv2.imshow(WINDOW, combo)

            # Handle click → map into left or right native coordinates
            if mouse["clicked"]:
                mouse["clicked"] = False
                mx, my = mouse["x"], mouse["y"]
                if my < combo.shape[0] - bar_h:
                    if mx < panel_split_x:
                        # left panel — scale from display to native
                        sx = frame_l.shape[1] / fl.shape[1]
                        sy = frame_l.shape[0] / fl.shape[0]
                        clicks.add("L", (int(mx * sx), int(my * sy)))
                    else:
                        fr = fit_height(disp_r, combo.shape[0])
                        sx = frame_r.shape[1] / fr.shape[1]
                        sy = frame_r.shape[0] / fr.shape[0]
                        lx = mx - panel_split_x
                        clicks.add("R", (int(lx * sx), int(my * sy)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("f"), ord("F")):
                clicks.mode = "far"
                print("Click mode: FAR post")
            elif key in (ord("n"), ord("N")):
                clicks.mode = "near"
                print("Click mode: NEAR post")
            elif key in (ord("c"), ord("C")):
                clicks.clear()
            elif key == ord("["):
                h_angle = max(0.0, h_angle - 1.0)
                print(f"  H_ANGLE → {h_angle:.1f}°")
            elif key == ord("]"):
                h_angle = min(180.0, h_angle + 1.0)
                print(f"  H_ANGLE → {h_angle:.1f}°")
            elif key == ord("-"):
                v_angle = max(0.0, v_angle - 1.0)
                print(f"  V_ANGLE → {v_angle:.1f}°")
            elif key == ord("="):
                v_angle = min(89.0, v_angle + 1.0)
                print(f"  V_ANGLE → {v_angle:.1f}°")
            elif key == ord(","):
                extend_up = max(1.0, extend_up - 0.5)
                print(f"  extend_up → {extend_up:.1f} m")
            elif key == ord("."):
                extend_up = min(20.0, extend_up + 0.5)
                print(f"  extend_up → {extend_up:.1f} m")
            elif key in (ord("r"), ord("R")):
                h_angle = float(cfg.H_ANGLE_DEG)
                v_angle = float(cfg.V_ANGLE_DEG)
                print(f"  Reset angles → H={h_angle:.1f} V={v_angle:.1f}")
            elif key in (ord("b"), ord("B")):
                ball_on = not ball_on
                print(f"  Ball detection {'ON' if ball_on else 'OFF'}")
            elif key in (ord("d"), ord("D")):
                show_labels = not show_labels
                print(f"  Landmark distance labels {'ON' if show_labels else 'OFF'}")
            elif key in (ord("s"), ord("S")):
                save_plane(
                    baseline, h_angle, v_angle, extend_up,
                    focal_px, img_w, img_h, clicks,
                )
            elif key in (ord("l"), ord("L")):
                data = load_plane()
                if data is None:
                    print("  No plane.json found")
                else:
                    h_angle = float(data.get("h_angle_deg", h_angle))
                    v_angle = float(data.get("v_angle_deg", v_angle))
                    extend_up = float(data.get("extend_up_m", extend_up))
                    clicks.load_dict(data.get("clicks", {}))
                    print(f"  Loaded {PLANE_PATH}  H={h_angle:.1f} V={v_angle:.1f}")

    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        if cap_l is not None:
            cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
