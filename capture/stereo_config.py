"""
stereo_config.py — Central configuration for the Hawkeye stereo rig.

Edit this file whenever the physical rig changes (baseline, angles, lens).
Stereo scripts read their defaults from here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Camera: B0332  ·  Lens: LN013 NOIR
  Datasheet at 1280×800 (sensor upright):  HFOV 70°  |  VFOV ≈47.3°
  Mounted SIDEWAYS on the posts → those axes swap in the image:
    Image left–right (width)  ≈ 47.3°
    Image up–down   (height)  ≈ 70°
  Focal-length tolerance: ±5 %  |  Distortion: 1.5 %
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rig geometry
────────────
  Cameras clamp to the post–crossbar corners, on the BACK of the posts
  (not in the plane of the wood). They aim across the goal toward the
  far post, tilted upward. Baseline is lens-to-lens along the bar.

    Baseline : 128 cm
    Behind   : ~10 cm  (lens centre behind the field-side face of the post)
    H-angle  : ~90°    (across the goal toward the far post)
    V-angle  : ~55°    (up from horizontal)

  With V_ANGLE=55° and image VFOV=70°, half-FOV=35° → lowest ray ≈ 20°
  above horizontal. Lowest visible point on the far post (horiz. dist ≈
  hypot(baseline, behind)):
    height  ≈ 0.47 m above the bar
    slant   ≈ 1.36 m  (ray length, not height)

World frame
───────────
  Origin : midpoint of the crossbar, on the goal plane (between the posts)
  X      : toward RIGHT post / right camera
  Y      : UP  (crossbar / cameras at Y = 0)
  Z      : into the field
  Goal plane : Z = 0  (posts + crossbar), extended upward for high balls
  Cameras    : Z = −CAM_BEHIND_POST_M  (behind the wood)
"""

from __future__ import annotations

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Physical rig ─────────────────────────────────────────────────────────────

BASELINE_M: float = 1.28       # centre-to-centre camera separation (metres)
# Lens centre behind the field-side face of the post (metres). Cameras clamp
# at the post–crossbar corner, on the back of the wood. Goal plane stays Z=0.
CAM_BEHIND_POST_M: float = 0.10
H_ANGLE_DEG: float = 90.0      # aim across goal toward far post (0° = into field)
V_ANGLE_DEG: float = 55.0      # tilt upward from horizontal (degrees)

# ── Datasheet (sensor upright, before sideways mount) ────────────────────────

DATASHEET_HFOV_DEG: float = 70.0      # along 1280-px axis when upright
DATASHEET_VFOV_DEG: float = 47.3      # along 800-px axis when upright
DATASHEET_WIDTH: int = 1280
DATASHEET_HEIGHT: int = 800
FOCAL_TOLERANCE_PCT: float = 5.0
DISTORTION_PCT: float = 1.5

# ── As mounted: cameras rotated 90° (sideways) ───────────────────────────────
# Image axes after sideways mount (rotate frames if the driver still outputs
# 1280×800 landscape — projection assumes image Y is downward on the sensor
# and world-up maps toward decreasing image Y when looking along cam_z).

CAMERAS_SIDEWAYS: bool = True
# Landscape frames (w > h) are rotated this way so the 70° FOV is vertical.
# 90 = clockwise, 270 = counter-clockwise, 0 = do not rotate.
# Skip when the frame is already portrait (h >= w).
SIDEWAYS_ROTATE_DEG: int = 90

# FOV in the CAPTURED image (swapped vs datasheet because of sideways mount)
HFOV_DEG: float = DATASHEET_VFOV_DEG   # ≈47.3° left–right in image
VFOV_DEG: float = DATASHEET_HFOV_DEG   # ≈70°   up–down in image

# Expected capture size when the stream is portrait after rotation
NATIVE_WIDTH: int = DATASHEET_HEIGHT   # 800
NATIVE_HEIGHT: int = DATASHEET_WIDTH   # 1280

# ── Derived focals (pin-hole; fx from width/HFOV, fy from height/VFOV) ────────

FOCAL_FX_SPEC: float = (NATIVE_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
FOCAL_FY_SPEC: float = (NATIVE_HEIGHT / 2.0) / math.tan(math.radians(VFOV_DEG / 2.0))
# Keep a single "spec" scalar for older call sites (prefer fx for width scaling)
FOCAL_PX_SPEC: float = FOCAL_FX_SPEC

# Lowest optical ray above horizontal: V_ANGLE − VFOV/2
LOWEST_RAY_DEG: float = V_ANGLE_DEG - (VFOV_DEG / 2.0)


def camera_positions(
    baseline_m: float = BASELINE_M,
    behind_m: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Left / right lens centres in world metres.

    Posts sit at X = ±baseline/2, Z = 0. Cameras share that X and Y (the
    post–crossbar corner) but sit behind the wood at Z = −behind.
    """
    d = CAM_BEHIND_POST_M if behind_m is None else behind_m
    half = baseline_m / 2.0
    return (-half, 0.0, -d), (half, 0.0, -d)


def lowest_visible_on_far_post(
    baseline_m: float = BASELINE_M,
    behind_m: float | None = None,
) -> tuple[float, float, float]:
    """
    Horizontal range to the far post is hypot(baseline, behind).
    Returns (lowest_ray_deg, height_on_post_m, slant_range_m).
    """
    d = CAM_BEHIND_POST_M if behind_m is None else behind_m
    horiz = math.hypot(baseline_m, d)
    alpha = math.radians(LOWEST_RAY_DEG)
    height = horiz * math.tan(alpha)
    slant = horiz / math.cos(alpha)
    return LOWEST_RAY_DEG, height, slant


def focal_axes(image_width: int, image_height: int) -> tuple[float, float]:
    """fx, fy in pixels from mounted FOVs and the actual frame size."""
    fx = (image_width / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    fy = (image_height / 2.0) / math.tan(math.radians(VFOV_DEG / 2.0))
    return fx, fy


def get_focal_px(image_width: int | None = None) -> tuple[float, str]:
    """Horizontal focal (fx) from the mounted image-HFOV and frame width."""
    w = image_width or NATIVE_WIDTH
    f = (w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    return f, f"spec-derived ({HFOV_DEG:.1f}° image-HFOV at {w} px wide)"


def print_summary(
    focal_px: float,
    focal_src: str,
    behind_m: float | None = None,
) -> None:
    """Print the active stereo rig config — called at startup of each script."""
    behind = CAM_BEHIND_POST_M if behind_m is None else behind_m
    ray, h_post, slant = lowest_visible_on_far_post(behind_m=behind)
    print(
        f"\n{'─'*60}\n"
        f"  Hawkeye stereo rig  (cameras SIDEWAYS)\n"
        f"{'─'*60}\n"
        f"  Datasheet : {DATASHEET_HFOV_DEG}° / {DATASHEET_VFOV_DEG}° at "
        f"{DATASHEET_WIDTH}×{DATASHEET_HEIGHT}\n"
        f"  In image  : HFOV {HFOV_DEG}°  VFOV {VFOV_DEG}°  "
        f"(70° is UP–DOWN)\n"
        f"  Native res: {NATIVE_WIDTH}×{NATIVE_HEIGHT} px  (portrait after rotate)\n"
        f"  Rotate    : {SIDEWAYS_ROTATE_DEG}° "
        f"{'(skip if already portrait)' if CAMERAS_SIDEWAYS else '(disabled)'}\n"
        f"  Baseline  : {BASELINE_M:.3f} m  (lens to lens)\n"
        f"  Behind    : {behind:.3f} m  (lens behind post face; plane at Z=0)\n"
        f"  H-angle   : {H_ANGLE_DEG:.1f}°  (across goal → far post)\n"
        f"  V-angle   : {V_ANGLE_DEG:.1f}°  (up from horizontal)\n"
        f"  Lowest ray: {ray:.1f}° above horizontal\n"
        f"  Far post  : lowest visible ≈ {h_post:.2f} m up the post  "
        f"(slant {slant:.2f} m)\n"
        f"  Focal fx  : {focal_px:.1f}  ← {focal_src}\n"
        f"  Spec fx/fy: {FOCAL_FX_SPEC:.1f} / {FOCAL_FY_SPEC:.1f}\n"
        f"  Posts zone: X ∈ [{-BASELINE_M/2:.3f}, {BASELINE_M/2:.3f}] m  Z = 0\n"
        f"{'─'*60}\n"
    )
