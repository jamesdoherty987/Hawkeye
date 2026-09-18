"""
stereo_plane_calib.py — Draw / verify the goal plane (test only).

Does NOT change stereo_record or stereo_calibrate. Use this to check that
the goal plane (between the posts, extended upward) projects correctly
into both camera views with your current stereo_config angles.

Assumptions
───────────
  Cameras clamp at the post–crossbar corners, on the BACK of the posts.
  Goal plane = posts + crossbar (Z = 0), extended UP for high balls.
  Left post  at X = −baseline/2,  right post at X = +baseline/2.
  Cameras share that X/Y but sit CAM_BEHIND_POST_M behind the wood (Z < 0).
  Crossbar height ≈ camera height (Y = 0).

  LEFT / RIGHT are from BEHIND the goal, looking into the field
  (goalkeeper's left/right). N on each camera points into the FIELD.

Controls
────────
  Click LEFT panel  = mark a point on the LEFT camera image
  Click RIGHT panel = mark a point on the RIGHT camera image

  F = next clicks are FAR post (default)
  K = lock the plane H-angle to those far-post clicks
  N = next clicks are NEAR post
  C = clear all click marks

  [ / ] = decrease / increase H_ANGLE by 1°
  - / = = decrease / increase V_ANGLE by 1°
  , / . = decrease / increase plane height extension (metres)
  R     = reset angles to stereo_config defaults
  B     = toggle ball detection (on by default)
          BLUE flash = ball seen, still in front of the plane
          GREEN      = ball has gone THROUGH between the posts
          RED        = ball has gone THROUGH but wide of the posts
  D     = toggle landmark distance labels on the plane
  S     = save plane overlay params → exports/stereo/plane.json
  L     = load plane.json if present
  Q/ESC = quit

  Video mode only:
  SPACE = pause / play
  A / D = back / forward ~1 second (both files stay in sync)
  Videos loop when either file ends.

  Image mode is a frozen pair (jpg/png/…). Same overlay tools; no seek.

  Sideways cameras: landscape frames (w > h) are rotated so the 70° FOV is
  vertical. Already-portrait files are left as-is. The full frame is always
  shown (letterboxed); the status bar sits under the pictures, not on them.
  O = cycle rotate 90° CW / 270° CCW / off

Usage
─────
  python capture/stereo_plane_calib.py --left 0 --right 1
  python capture/stereo_plane_calib.py --video-left left.mp4 --video-right right.mp4
  python capture/stereo_plane_calib.py --image-left left.jpg --image-right right.jpg
  python capture/stereo_plane_calib.py --image-left left.jpg --image-right right.jpg --rotate 270
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
from stereo_calibrate import (
    make_proj_matrix,
    project_world,
    triangulate_3d,
    calib_centres_from_dets,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANE_PATH = PROJECT_ROOT / "exports" / "stereo" / "plane.json"
STILL_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32
IMAGE_SIZE = 640
CONFIDENCE = 0.30
WINDOW = "Hawkeye Plane"

PLANE_CROSS_M = 0.05
POST_MARGIN_M = 0.02          # extra width beyond ±baseline/2 still called "between"
REPROJ_MAX_PX = 40.0          # reject 3D if it doesn't land on both detections
CALL_LATCH_S = 1.2
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
COL_CALL_BLUE = (255, 90, 20)     # BGR — in field, not yet through
COL_CALL_GREEN = (0, 255, 80)     # through + between posts
COL_CALL_RED = (0, 0, 255)        # through + wide


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


def resolve_media_path(raw: str, kind: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise RuntimeError(f"{kind} not found: {p}")
    return p


def is_still_path(path: Path) -> bool:
    return path.suffix.lower() in STILL_EXTS


def orient_frame(frame: np.ndarray, rotate_deg: int) -> np.ndarray:
    """
    Rotate landscape captures to portrait for the sideways mount.
    Already-portrait frames (height >= width) are unchanged.
    rotate_deg: 90 clockwise, 270 counter-clockwise, 0 skip.
    """
    if not cfg.CAMERAS_SIDEWAYS or rotate_deg == 0:
        return frame
    h, w = frame.shape[:2]
    if h >= w:
        return frame
    if rotate_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def rotate_label(deg: int) -> str:
    if not cfg.CAMERAS_SIDEWAYS or deg == 0:
        return "rotate=off"
    if deg == 270:
        return "rotate=270 CCW"
    return "rotate=90 CW"


def draw_camera_hud(img: np.ndarray, which_cam: str, rotate_deg: int) -> None:
    col = COL_POST if which_cam == "L" else COL_CLICK_NEAR
    put_label(img, "L" if which_cam == "L" else "R", (8, 26), col, 0.75, 2)
    h, w = img.shape[:2]
    if h < w:
        put_label(img, "O rotate", (8, h - 14), (0, 0, 255), 0.5, 2)


def load_still(path: Path) -> np.ndarray:
    """Read a still; imdecode handles Windows paths that cv2.imread can miss."""
    img: np.ndarray | None = None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    print(f"  {path.name} — {w}×{h}  still")
    return img


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Could not read frames from: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h, w = frame.shape[:2]
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"  {path.name} — {w}×{h}  {n} frames  {fps:.1f} fps")
    return cap


def video_frame_count(cap: cv2.VideoCapture) -> int:
    return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))


def video_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1.0 else 30.0


def seek_videos(
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
    frame_idx: int,
    n_frames: int,
) -> int:
    """Seek both files to the same frame so they stay paired."""
    if n_frames <= 0:
        return 0
    idx = int(max(0, min(frame_idx, n_frames - 1)))
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, idx)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return idx


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
    y_vis_min: float = 0.0,
) -> None:
    """Draw the visible part of the goal plane. The bar is often below the FOV."""
    half = baseline_m / 2.0
    y0 = max(0.0, float(y_vis_min))
    y1 = max(y0 + 0.2, float(extend_up_m))

    # Visible uprights (from lowest ray up — not from the unseen bar)
    draw_line_world(img, P, np.array([-half, y0, 0.0]), np.array([-half, y1, 0.0]), COL_POST, 3)
    draw_line_world(img, P, np.array([half, y0, 0.0]), np.array([half, y1, 0.0]), COL_POST, 3)
    draw_line_world(img, P, np.array([-half, y0, 0.0]), np.array([half, y0, 0.0]), COL_BAR, 2)
    draw_line_world(img, P, np.array([-half, y1, 0.0]), np.array([half, y1, 0.0]), COL_GRID, 1)

    for i in range(grid_nx + 1):
        x = -half + (baseline_m * i / grid_nx)
        a = np.array([x, y0, 0.0])
        b = np.array([x, y1, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=16)

    for j in range(1, grid_ny + 1):
        y = y0 + (y1 - y0) * j / grid_ny
        a = np.array([-half, y, 0.0])
        b = np.array([half, y, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=16)

    uv = project_safe(P, np.array([0.0, y0, 0.0]))
    if uv is not None:
        put_label(img, f"FOV from {y0:.2f}m above bar", (uv[0] + 6, uv[1] + 16), COL_DEBUG, 0.4, 1)

    if not show_point_labels:
        return

    far_name = "R" if which_cam == "L" else "L"
    landmarks: list[tuple[str, np.ndarray]] = [
        (f"{far_name}@{y0:.2f}m", np.array([half if far_name == "R" else -half, y0, 0.0])),
        (f"{far_name}-mid", np.array([half if far_name == "R" else -half, (y0 + y1) * 0.5, 0.0])),
        (f"{far_name}-top", np.array([half if far_name == "R" else -half, y1, 0.0])),
    ]
    for name, xyz in landmarks:
        label_world_point(img, P, xyz, pos_l, pos_r, name, which_cam)


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if target_h <= 0 or img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)


def side_by_side_full(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Stack both views at a common height. Uniform scale, no crop."""
    h = max(left.shape[0], right.shape[0])
    fl = fit_height(left, h)
    fr = fit_height(right, h)
    return np.hstack([fl, fr]), fl.shape[1]


def attach_status_bar(
    img: np.ndarray,
    lines: list[str],
    accent_color: tuple[int, int, int] | None = None,
    accent_line: str = "",
) -> tuple[np.ndarray, int]:
    """Append a status strip under the image so UI never covers the picture."""
    bar_h = 20 * max(len(lines), 1) + 12
    bar = np.zeros((bar_h, img.shape[1], 3), dtype=img.dtype)
    for i, line in enumerate(lines):
        col = accent_color if (accent_line and line == accent_line) else COL_STATUS
        cv2.putText(
            bar, line, (10, 18 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA,
        )
    return np.vstack([img, bar]), bar_h


def window_client_size() -> tuple[int, int] | None:
    try:
        rect = cv2.getWindowImageRect(WINDOW)
        if rect is not None and len(rect) >= 4 and int(rect[2]) > 16 and int(rect[3]) > 16:
            return int(rect[2]), int(rect[3])
    except cv2.error:
        pass
    return None


def letterbox(img: np.ndarray, win_w: int, win_h: int) -> tuple[np.ndarray, int, int, int, int]:
    """
    Fit the whole image into the window with black bars. No crop, no stretch.
    Returns (canvas, x0, y0, content_w, content_h).
    """
    ih, iw = img.shape[:2]
    if win_w <= 0 or win_h <= 0 or iw <= 0 or ih <= 0:
        return img, 0, 0, iw, ih
    scale = min(win_w / iw, win_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    if (nw, nh) == (iw, ih):
        resized = img
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((win_h, win_w, 3), dtype=img.dtype)
    x0 = (win_w - nw) // 2
    y0 = (win_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, x0, y0, nw, nh


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
        self.far_l = [(int(p[0]), int(p[1])) for p in data.get("far_left", [])]
        self.far_r = [(int(p[0]), int(p[1])) for p in data.get("far_right", [])]
        self.near_l = [(int(p[0]), int(p[1])) for p in data.get("near_left", [])]
        self.near_r = [(int(p[0]), int(p[1])) for p in data.get("near_right", [])]


# ─── Ball (YOLO + triangulation vs the Z=0 plane) ────────────────────────────

def default_ball_model() -> str:
    custom = PROJECT_ROOT / "models" / "football_yolov8n.pt"
    return str(custom) if custom.exists() else COCO_MODEL


def load_ball_model(model_str: str) -> tuple[YOLO, list[int] | None]:
    path = Path(model_str)
    if not path.is_absolute():
        full = PROJECT_ROOT / path
        if full.exists():
            path = full
    is_coco = path.name == COCO_MODEL
    print(f"Loading {'COCO sports-ball' if is_coco else 'custom'} model: {path}")
    return YOLO(str(path)), ([SPORTS_BALL_CLASS] if is_coco else None)


def detect_both(model: YOLO, frame_l, frame_r, conf: float, classes: list[int] | None):
    kw: dict = dict(conf=conf, imgsz=IMAGE_SIZE, verbose=False)
    if classes:
        kw["classes"] = classes
    results = model.predict([frame_l, frame_r], **kw)

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


def plane_side_label(z: float, eps: float = PLANE_CROSS_M) -> str:
    if z > eps:
        return "field"
    if z < -eps:
        return "through"
    return "on-plane"


def point_to_post_px(
    P: np.ndarray,
    click: tuple[int, int],
    post_x: float,
    y0: float,
    y1: float,
    n: int = 32,
) -> float:
    """Pixel distance from a click to the projected far-post line."""
    best = 1e9
    cx, cy = float(click[0]), float(click[1])
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i / n
        uv = project_safe(P, np.array([post_x, y, 0.0]))
        if uv is None:
            continue
        best = min(best, math.hypot(uv[0] - cx, uv[1] - cy))
    return best


def lock_h_to_far_clicks(
    clicks: ClickState,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    h_angle: float,
    v_angle: float,
    focal_l: float,
    focal_r: float,
    img_w: int,
    img_h: int,
    img_w_r: int,
    img_h_r: int,
    baseline: float,
    y0: float,
    y1: float,
) -> tuple[float, float]:
    """
    Sweep H-angle so the drawn far post sits on the latest FAR clicks.
    Left click → right post; right click → left post.
    """
    half = baseline / 2.0
    pt_l = clicks.far_l[-1] if clicks.far_l else None
    pt_r = clicks.far_r[-1] if clicks.far_r else None
    if pt_l is None and pt_r is None:
        return h_angle, -1.0

    lo = max(1.0, h_angle - 20.0)
    hi = min(90.0, h_angle + 20.0)
    best_h = h_angle
    best_err = 1e9
    step = 0.5
    h = lo
    while h <= hi + 1e-9:
        P1 = make_proj_matrix(pos_l, +h, v_angle, focal_l, img_w, img_h)
        P2 = make_proj_matrix(pos_r, -h, v_angle, focal_r, img_w_r, img_h_r)
        err = 0.0
        n = 0
        if pt_l is not None:
            err += point_to_post_px(P1, pt_l, +half, y0, y1)
            n += 1
        if pt_r is not None:
            err += point_to_post_px(P2, pt_r, -half, y0, y1)
            n += 1
        mean = err / max(n, 1)
        if mean < best_err:
            best_err = mean
            best_h = h
        h += step
    return best_h, best_err


def reproj_err_px(P: np.ndarray, xyz: np.ndarray, uv: tuple[float, float]) -> float:
    pred = project_world(P, xyz)
    if pred is None:
        return 1e9
    return math.hypot(pred[0] - uv[0], pred[1] - uv[1])


def inside_posts(x: float, baseline: float, radius: float = 0.0) -> bool:
    """True only if the whole ball (centre ± radius) sits between the posts."""
    return abs(x) + radius <= baseline / 2.0 + POST_MARGIN_M


def live_ball_call(
    x: float,
    z: float,
    baseline: float,
    radius: float = 0.0,
) -> str:
    """
    Score uses the 3D centre, expanded by ball radius.
    A 22 cm ball on a 1 m goal is IN only if |X| + 0.11 m ≤ 0.50 m.
    """
    in_posts = inside_posts(x, baseline, radius)
    if z > PLANE_CROSS_M:
        return "approaching" if in_posts else "wide_field"
    if in_posts:
        return "goal"
    return "wide"


def update_call_latch(
    call: str,
    z: float | None,
    now: float,
    state: dict,
) -> str:
    """Latch GOAL/WIDE briefly after a crossing, but never if the ball is gone or 1-cam."""
    if call in ("none", "one", "reject"):
        if call == "none":
            state["prev_z"] = None
            state["latch"] = None
        return call

    prev_z = state.get("prev_z")
    if (
        z is not None
        and prev_z is not None
        and prev_z > PLANE_CROSS_M
        and z <= PLANE_CROSS_M
        and call in ("goal", "wide")
    ):
        state["latch"] = call
        state["latch_until"] = now + CALL_LATCH_S
        print(f"  Plane crossed → {call.upper()}")
    state["prev_z"] = z

    # A later WIDE reading must not keep a stale GOAL flash
    if call == "wide":
        state["latch"] = "wide"
        return "wide"

    latch = state.get("latch")
    until = float(state.get("latch_until") or 0.0)
    if latch in ("goal", "wide") and now < until:
        return latch
    return call


def call_style(call: str) -> tuple[tuple[int, int, int], str, bool]:
    """colour, short label, pulse. Empty label = do not highlight."""
    if call == "approaching":
        return COL_CALL_BLUE, "IN FIELD", True
    if call == "wide_field":
        return COL_CALL_BLUE, "WIDE", True
    if call == "goal":
        return COL_CALL_GREEN, "GOAL", False
    if call == "wide":
        return COL_CALL_RED, "WIDE", False
    return (80, 80, 80), "", False


def draw_flash_border(img: np.ndarray, color: tuple[int, int, int], now: float, pulse: bool) -> None:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return
    if pulse:
        thick = 6 + int(6 * (0.5 + 0.5 * math.sin(now * 14.0)))
    else:
        thick = 10
    thick = max(4, min(thick, min(h, w) // 8))
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), color, thick)


def paint_ball_box(img: np.ndarray, det, color: tuple[int, int, int], label: str) -> None:
    if det is None:
        return
    x1, y1, x2, y2 = map(int, det[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.circle(img, (int(det[0]), int(det[1])), 5, color, -1)
    if label:
        put_label(img, label, (x1, max(20, y1 - 8)), color, 0.48, 2)


def paint_ball_pair(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    det_l,
    det_r,
    P1: np.ndarray,
    P2: np.ndarray,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    baseline: float,
    now: float,
    latch_state: dict,
) -> tuple[str, str]:
    """Draw the ball AFTER the grid. No coloured frame unless 3D from BOTH cameras."""
    if det_l is None and det_r is None:
        update_call_latch("none", None, now, latch_state)
        return "", "none"

    if det_l is None or det_r is None:
        update_call_latch("one", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        which = "L only" if det_r is None else "R only"
        half = baseline / 2.0
        return f"dbg  {which}  (need both for 3D)  posts ±{half:.2f}m", "one"

    pt_l, pt_r, _corr, _rough = calib_centres_from_dets(
        det_l, det_r, pos_l, pos_r, P1, P2,
    )
    pos3d = triangulate_3d(P1, P2, pt_l, pt_r)
    if pos3d is None:
        update_call_latch("reject", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        return "dbg  both seen  3D failed", "reject"

    x, y, z = map(float, pos3d)
    err_l = reproj_err_px(P1, pos3d, pt_l)
    err_r = reproj_err_px(P2, pos3d, pt_r)
    if err_l > REPROJ_MAX_PX or err_r > REPROJ_MAX_PX:
        update_call_latch("reject", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        return (
            f"dbg  3D mismatch  reproj L{err_l:.0f}px R{err_r:.0f}px  "
            f"(ignore call)"
        ), "reject"

    radius = cfg.BALL_RADIUS_M
    raw = live_ball_call(x, z, baseline, radius)
    call = update_call_latch(raw, z, now, latch_state)
    color, banner, pulse = call_style(call)
    if banner:
        paint_ball_box(disp_l, det_l, color, banner)
        paint_ball_box(disp_r, det_r, color, banner)
        draw_flash_border(disp_l, color, now, pulse)
        draw_flash_border(disp_r, color, now, pulse)
    else:
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")

    half = baseline / 2.0
    in_posts = inside_posts(x, baseline, radius)
    x0, x1 = x - radius, x + radius
    line = (
        f"dbg  X={x:+.2f} ball[{x0:+.2f},{x1:+.2f}] posts±{half:.2f} "
        f"{'IN' if in_posts else 'OUT'}  "
        f"Y={y:+.2f}  Z={z:+.2f} ({plane_side_label(z)})  "
        f"reproj {err_l:.0f}/{err_r:.0f}px"
        + (f"  {banner}" if banner else "")
    )
    return line, call


# ─── Save / load ─────────────────────────────────────────────────────────────

def save_plane(
    baseline_m: float,
    behind_m: float,
    h_angle: float,
    v_angle: float,
    extend_up_m: float,
    focal_px: float,
    img_w: int,
    img_h: int,
    clicks: ClickState,
    rotate_deg: int,
) -> None:
    PLANE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_m": baseline_m,
        "cam_behind_post_m": behind_m,
        "h_angle_deg": h_angle,
        "v_angle_deg": v_angle,
        "extend_up_m": extend_up_m,
        "focal_px": focal_px,
        "image_width": img_w,
        "image_height": img_h,
        "cameras_sideways": cfg.CAMERAS_SIDEWAYS,
        "rotate_deg": rotate_deg,
        "plane": {
            "origin": "midpoint of crossbar on the goal plane",
            "equation": "Z = 0 (goal plane / posts / crossbar)",
            "cameras_z": -behind_m,
            "left_post_x": -baseline_m / 2.0,
            "right_post_x": baseline_m / 2.0,
            "crossbar_y": 0.0,
            "note": "Y > 0 is above the bar; Z > 0 is into the field; cameras at Z = −behind",
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

def _require_both(left, right, name: str) -> bool | None:
    """True if both set, False if neither, None if only one (invalid)."""
    if left and right:
        return True
    if left or right:
        print(f"ERROR: provide BOTH --{name}-left and --{name}-right.", file=sys.stderr)
        return None
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Goal-plane overlay / click calibration test")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument(
        "--video-left", type=str, default=None,
        help="Left-camera video (use with --video-right instead of live cameras)",
    )
    parser.add_argument(
        "--video-right", type=str, default=None,
        help="Right-camera video (use with --video-left instead of live cameras)",
    )
    parser.add_argument(
        "--image-left", type=str, default=None,
        help="Left-camera still (jpg/png/…). Use with --image-right",
    )
    parser.add_argument(
        "--image-right", type=str, default=None,
        help="Right-camera still (jpg/png/…). Use with --image-left",
    )
    parser.add_argument("--baseline", type=float, default=cfg.BASELINE_M)
    parser.add_argument(
        "--behind", type=float, default=cfg.CAM_BEHIND_POST_M,
        help="Lens centre behind the field-side post face (metres)",
    )
    parser.add_argument("--h-angle", type=float, default=cfg.H_ANGLE_DEG)
    parser.add_argument("--v-angle", type=float, default=cfg.V_ANGLE_DEG)
    parser.add_argument("--extend", type=float, default=4.0,
                        help="How far above the bar to draw the plane (metres)")
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument(
        "--rotate", type=int, default=None, choices=(0, 90, 270),
        help="Rotate landscape frames: 90=CW, 270=CCW, 0=off "
             f"(default {cfg.SIDEWAYS_ROTATE_DEG} from stereo_config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ball detector. Default: models/football_yolov8n.pt if present, else yolov8n.pt",
    )
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--no-ball", action="store_true", help="Start with ball detection off (B toggles)")
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

    if (args.video_left or args.video_right) and (args.image_left or args.image_right):
        print("ERROR: use either --video-* or --image-*, not both.", file=sys.stderr)
        return 1
    want_video = _require_both(args.video_left, args.video_right, "video")
    want_image = _require_both(args.image_left, args.image_right, "image")
    if want_video is None or want_image is None:
        return 1
    if not want_video and not want_image and args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1

    source = "live"
    if want_image:
        source = "still"
    elif want_video:
        source = "video"

    h_angle = float(args.h_angle)
    v_angle = float(args.v_angle)
    extend_up = float(args.extend)
    baseline = float(args.baseline)
    behind = float(args.behind)
    if baseline <= 0:
        print("ERROR: --baseline must be > 0", file=sys.stderr)
        return 1
    if behind < 0:
        print("ERROR: --behind must be >= 0", file=sys.stderr)
        return 1
    rotate_deg = int(args.rotate) if args.rotate is not None else int(cfg.SIDEWAYS_ROTATE_DEG)
    if rotate_deg not in (0, 90, 270):
        rotate_deg = 90
    clicks = ClickState()
    ball_on = not bool(args.no_ball)
    show_labels = False
    paused = source != "live"
    model: YOLO | None = None
    ball_classes: list[int] | None = None
    ball_dets: tuple | None = None
    ball_latch: dict = {"prev_z": None, "latch": None, "latch_until": 0.0}
    panel_split_x = 0
    frame_i = 0
    fps_ema = 0.0
    last_t = time.time()
    video_n = 0
    play_fps = 30.0
    src_l = f"cam {args.left}"
    src_r = f"cam {args.right}"

    mouse = {"x": 0, "y": 0, "clicked": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["x"] = x
            mouse["y"] = y
            mouse["clicked"] = True

    cap_l = cap_r = None
    frame_l: np.ndarray | None = None
    frame_r: np.ndarray | None = None
    try:
        if source == "still":
            path_l = resolve_media_path(args.image_left, "Image")
            path_r = resolve_media_path(args.image_right, "Image")
            print(f"\nOpening left still:  {path_l}")
            frame_l = load_still(path_l)
            print(f"Opening right still: {path_r}")
            frame_r = load_still(path_r)
            src_l, src_r = path_l.name, path_r.name
            probe_l, probe_r = frame_l, frame_r
        elif source == "video":
            path_l = resolve_media_path(args.video_left, "Video")
            path_r = resolve_media_path(args.video_right, "Video")
            if is_still_path(path_l) or is_still_path(path_r):
                if not (is_still_path(path_l) and is_still_path(path_r)):
                    print(
                        "ERROR: mix of still and video. Use two images or two videos.",
                        file=sys.stderr,
                    )
                    return 1
                source = "still"
                paused = True
                print(f"\nOpening left still:  {path_l}")
                frame_l = load_still(path_l)
                print(f"Opening right still: {path_r}")
                frame_r = load_still(path_r)
                src_l, src_r = path_l.name, path_r.name
                probe_l, probe_r = frame_l, frame_r
            else:
                print(f"\nOpening left video:  {path_l}")
                cap_l = open_video(path_l)
                print(f"Opening right video: {path_r}")
                cap_r = open_video(path_r)
                src_l, src_r = path_l.name, path_r.name
                video_n = min(video_frame_count(cap_l), video_frame_count(cap_r))
                play_fps = min(video_fps(cap_l), video_fps(cap_r))
                ok_l, probe_l = read_cam(cap_l)
                ok_r, probe_r = read_cam(cap_r)
                if not ok_l or probe_l is None or not ok_r or probe_r is None:
                    print("ERROR: could not read initial video frames.", file=sys.stderr)
                    return 1
                seek_videos(cap_l, cap_r, 0, video_n)
                paused = False
        else:
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

        view_probe_l = orient_frame(probe_l, rotate_deg)
        view_probe_r = orient_frame(probe_r, rotate_deg)
        img_w, img_h = view_probe_l.shape[1], view_probe_l.shape[0]
        img_w_r, img_h_r = view_probe_r.shape[1], view_probe_r.shape[0]
        raw_l_wh = (probe_l.shape[1], probe_l.shape[0])
        raw_r_wh = (probe_r.shape[1], probe_r.shape[0])
        if raw_l_wh != (img_w, img_h) or raw_r_wh != (img_w_r, img_h_r):
            print(
                f"  Oriented landscape → portrait ({rotate_label(rotate_deg)}): "
                f"L {raw_l_wh[0]}×{raw_l_wh[1]}→{img_w}×{img_h}  "
                f"R {raw_r_wh[0]}×{raw_r_wh[1]}→{img_w_r}×{img_h_r}"
            )
        elif rotate_deg and cfg.CAMERAS_SIDEWAYS:
            print(
                f"  Frames already portrait — no rotate applied "
                f"(O still cycles {rotate_label(rotate_deg)})"
            )
        if (img_w, img_h) != (img_w_r, img_h_r):
            print(
                f"WARNING: left is {img_w}×{img_h}, right is {img_w_r}×{img_h_r}. "
                "Projection uses each view's own size."
            )
        if args.focal is not None:
            focal_px, focal_src = float(args.focal), "CLI"
            focal_r = focal_px * (img_w_r / img_w) if img_w else focal_px
        else:
            focal_px, focal_src = cfg.get_focal_px(img_w)
            focal_r, _ = cfg.get_focal_px(img_w_r)

        cfg.print_summary(focal_px, focal_src, behind_m=behind)
        extra = ""
        if source == "video":
            extra = (
                f"Video replay: {src_l} | {src_r}  ({video_n} frames @ {play_fps:.1f} fps)\n"
                "SPACE pause  A/D seek  (files loop and stay in sync)\n"
            )
        elif source == "still":
            extra = f"Still pair: {src_l} | {src_r}\n"
        print(
            f"Posts {baseline:.2f} m apart.  L/R = behind the goal, facing the field.\n"
            f"{extra}"
            "Blue = in field (whole ball between posts).  Green = through + between.  "
            "Red = through but wide.  Need both cameras.\n"
            "Click the visible FAR post then press K to lock the plane to it.\n"
        )

        xyz_l, xyz_r = cfg.camera_positions(baseline, behind)
        pos_l = np.array(xyz_l)
        pos_r = np.array(xyz_r)
        post_l = np.array([-baseline / 2.0, 0.0, 0.0])
        post_r = np.array([baseline / 2.0, 0.0, 0.0])

        def build_projs():
            # Left looks across toward +X (far/right post); right toward −X
            P1 = make_proj_matrix(pos_l, +h_angle, v_angle, focal_px, img_w, img_h)
            P2 = make_proj_matrix(pos_r, -h_angle, v_angle, focal_r, img_w_r, img_h_r)
            return P1, P2

        P1, P2 = build_projs()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 900)
        cv2.setMouseCallback(WINDOW, on_mouse)
        layout = {
            "x0": 0, "y0": 0, "nw": 1, "nh": 1,
            "src_w": 1, "src_h": 1, "split_x": 0, "bar_h": 0, "pair_h": 1,
        }

        while True:
            if source != "still" and (not paused or frame_l is None or frame_r is None):
                assert cap_l is not None and cap_r is not None
                ok_l, next_l = read_cam(cap_l)
                ok_r, next_r = read_cam(cap_r)
                if source == "video" and (not ok_l or next_l is None or not ok_r or next_r is None):
                    seek_videos(cap_l, cap_r, 0, video_n)
                    ok_l, next_l = read_cam(cap_l)
                    ok_r, next_r = read_cam(cap_r)
                    ball_dets = None
                if not ok_l or next_l is None or not ok_r or next_r is None:
                    time.sleep(0.02)
                    continue
                frame_l, frame_r = next_l, next_r
                if source == "live" or not paused:
                    ball_dets = None

            if frame_l is None or frame_r is None:
                time.sleep(0.02)
                continue

            view_l = orient_frame(frame_l, rotate_deg)
            view_r = orient_frame(frame_r, rotate_deg)
            img_w, img_h = view_l.shape[1], view_l.shape[0]
            img_w_r, img_h_r = view_r.shape[1], view_r.shape[0]
            if args.focal is not None:
                focal_px = float(args.focal)
                focal_r = focal_px * (img_w_r / img_w) if img_w else focal_px
            else:
                focal_px, _ = cfg.get_focal_px(img_w)
                focal_r, _ = cfg.get_focal_px(img_w_r)

            # Rebuild projection if angles / size / rotate changed
            P1, P2 = build_projs()

            now = time.time()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps_ema = (1.0 / dt) if frame_i == 0 else (0.9 * fps_ema + 0.1 / dt)
            frame_i += 1

            disp_l = view_l.copy()
            disp_r = view_r.copy()
            _, y_vis, _ = cfg.lowest_visible_on_far_post(baseline, behind, v_angle)
            y_vis = max(0.0, float(y_vis))
            draw_goal_plane(
                disp_l, P1, baseline, extend_up, pos_l, pos_r, "L",
                show_point_labels=show_labels, y_vis_min=y_vis,
            )
            draw_goal_plane(
                disp_r, P2, baseline, extend_up, pos_l, pos_r, "R",
                show_point_labels=show_labels, y_vis_min=y_vis,
            )
            click_dbg = clicks.draw_on(disp_l, disp_r, P1, P2, pos_l, pos_r)

            ball_line = ""
            ball_call = "none"
            if ball_on:
                if model is None:
                    model, ball_classes = load_ball_model(args.model or default_ball_model())
                if ball_dets is None:
                    ball_dets = detect_both(
                        model, view_l, view_r, args.conf, ball_classes,
                    )
                ball_line, ball_call = paint_ball_pair(
                    disp_l, disp_r, ball_dets[0], ball_dets[1],
                    P1, P2, pos_l, pos_r, baseline, now, ball_latch,
                )

            draw_camera_hud(disp_l, "L", rotate_deg)
            draw_camera_hud(disp_r, "R", rotate_deg)

            combo, panel_split_x = side_by_side_full(disp_l, disp_r)
            color, banner, pulse = call_style(ball_call)
            if banner:
                draw_flash_border(combo, color, now, pulse)
            pair_h = combo.shape[0]

            src_note = ""
            if source == "video" and cap_l is not None:
                nxt = int(cap_l.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                shown = max(1, nxt)
                src_note = f"  {shown}/{video_n}"
            lines = [
                f"L {src_l}  R {src_r}  |  posts {baseline:.2f}m  "
                f"ball Ø{cfg.BALL_DIAMETER_M*100:.0f}cm  "
                f"H{h_angle:.0f} V{v_angle:.0f}  visY≥{y_vis:.2f}m  "
                f"{fps_ema:.0f}fps  {rotate_label(rotate_deg)}{src_note}",
            ]
            if ball_line:
                lines.append(ball_line)
            lines.extend(click_dbg[:1])
            packed, bar_h = attach_status_bar(
                combo, lines,
                accent_color=color if banner else None,
                accent_line=ball_line,
            )

            win = window_client_size() or (1600, 900)
            canvas, x0, y0, nw, nh = letterbox(packed, win[0], win[1])
            layout.update({
                "x0": x0, "y0": y0, "nw": nw, "nh": nh,
                "src_w": packed.shape[1], "src_h": packed.shape[0],
                "split_x": panel_split_x, "bar_h": bar_h, "pair_h": pair_h,
            })
            cv2.imshow(WINDOW, canvas)

            if mouse["clicked"]:
                mouse["clicked"] = False
                mx, my = mouse["x"], mouse["y"]
                nw, nh = layout["nw"], layout["nh"]
                if nw > 0 and nh > 0:
                    x0, y0 = layout["x0"], layout["y0"]
                    if x0 <= mx < x0 + nw and y0 <= my < y0 + nh:
                        sx = (mx - x0) * layout["src_w"] / nw
                        sy = (my - y0) * layout["src_h"] / nh
                        pair_h = layout["pair_h"]
                        split_x = layout["split_x"]
                        if 0 <= sy < pair_h:
                            if sx < split_x and split_x > 0:
                                x = max(0, min(view_l.shape[1] - 1, int(sx * view_l.shape[1] / split_x)))
                                y = max(0, min(view_l.shape[0] - 1, int(sy * view_l.shape[0] / pair_h)))
                                clicks.add("L", (x, y))
                            elif layout["src_w"] > split_x:
                                lx = sx - split_x
                                rw = layout["src_w"] - split_x
                                x = max(0, min(view_r.shape[1] - 1, int(lx * view_r.shape[1] / rw)))
                                y = max(0, min(view_r.shape[0] - 1, int(sy * view_r.shape[0] / pair_h)))
                                clicks.add("R", (x, y))

            if source == "video":
                delay = 30 if paused else max(1, int(round(1000.0 / play_fps)))
            elif source == "still":
                delay = 30
            else:
                delay = 1
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif source == "video" and key == 32:  # SPACE
                paused = not paused
                print(f"  Video {'PAUSED' if paused else 'PLAY'}")
            elif source == "video" and cap_l is not None and cap_r is not None and key in (
                ord("a"), ord("A"), ord("d"), ord("D"),
            ):
                step = max(1, int(round(play_fps)))
                nxt = int(cap_l.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                current = max(0, nxt - 1)
                target = current - step if key in (ord("a"), ord("A")) else current + step
                seek_videos(cap_l, cap_r, target, video_n)
                frame_l = frame_r = None
                ball_dets = None
                print(f"  Seek → frame {max(0, min(target, max(video_n - 1, 0)))}/{video_n}")
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
            elif key in (ord("k"), ord("K")):
                if not clicks.far_l and not clicks.far_r:
                    print("  K: click the FAR post in one or both views first (F mode)")
                else:
                    new_h, err = lock_h_to_far_clicks(
                        clicks, pos_l, pos_r, h_angle, v_angle,
                        focal_px, focal_r, img_w, img_h, img_w_r, img_h_r,
                        baseline, y_vis, extend_up,
                    )
                    h_angle = new_h
                    print(
                        f"  Locked H → {h_angle:.1f}°  "
                        f"(far-post click err {err:.0f}px). R resets."
                    )
            elif key in (ord("r"), ord("R")):
                h_angle = float(cfg.H_ANGLE_DEG)
                v_angle = float(cfg.V_ANGLE_DEG)
                print(f"  Reset angles → H={h_angle:.1f} V={v_angle:.1f}")
            elif key in (ord("b"), ord("B")):
                ball_on = not ball_on
                ball_dets = None
                ball_latch["prev_z"] = None
                ball_latch["latch"] = None
                print(f"  Ball detection {'ON' if ball_on else 'OFF'}")
            elif key in (ord("d"), ord("D")):
                show_labels = not show_labels
                print(f"  Landmark distance labels {'ON' if show_labels else 'OFF'}")
            elif key in (ord("o"), ord("O")):
                rotate_deg = {90: 270, 270: 0, 0: 90}.get(rotate_deg, 90)
                clicks.clear()
                ball_dets = None
                print(f"  {rotate_label(rotate_deg)}  (clicks cleared)")
            elif key in (ord("s"), ord("S")):
                save_plane(
                    baseline, behind, h_angle, v_angle, extend_up,
                    focal_px, img_w, img_h, clicks, rotate_deg,
                )
            elif key in (ord("l"), ord("L")):
                data = load_plane()
                if data is None:
                    print("  No plane.json found")
                else:
                    h_angle = float(data.get("h_angle_deg", h_angle))
                    v_angle = float(data.get("v_angle_deg", v_angle))
                    extend_up = float(data.get("extend_up_m", extend_up))
                    behind = float(data.get("cam_behind_post_m", behind))
                    xyz_l, xyz_r = cfg.camera_positions(baseline, behind)
                    pos_l = np.array(xyz_l)
                    pos_r = np.array(xyz_r)
                    rd = int(data.get("rotate_deg", rotate_deg))
                    if rd in (0, 90, 270) and rd != rotate_deg:
                        rotate_deg = rd
                        ball_dets = None
                    clicks.load_dict(data.get("clicks", {}))
                    print(
                        f"  Loaded {PLANE_PATH}  H={h_angle:.1f} V={v_angle:.1f}  "
                        f"behind={behind:.3f} m  {rotate_label(rotate_deg)}"
                    )

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
