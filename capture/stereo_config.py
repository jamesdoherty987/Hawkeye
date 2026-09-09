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
  Cameras at the post–crossbar corners, aimed across the goal toward the
  far post, tilted upward. Baseline is camera-to-camera along the bar.

    Baseline : 128 cm
    H-angle  : ~90°   (across the goal toward the far post)
    V-angle  : ~55°   (up from horizontal)

  With V_ANGLE=55° and image VFOV=70°, half-FOV=35° → lowest ray ≈ 20°
  above horizontal. Lowest visible point on the far post (horiz. dist = baseline):
    height  = baseline × tan(20°)  ≈ 0.47 m above the bar
    slant   = baseline / cos(20°) ≈ 1.36 m  (ray length, not height)

World frame
───────────
  Origin : midpoint between cameras
  X      : toward RIGHT camera
  Y      : UP
  Z      : into the field
  Goal plane : Z = 0  (posts + crossbar), extended upward for high balls
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Physical rig ─────────────────────────────────────────────────────────────

BASELINE_M: float = 1.28       # centre-to-centre camera separation (metres)
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


def lowest_visible_on_far_post(baseline_m: float = BASELINE_M) -> tuple[float, float, float]:
    """
    Far post is `baseline_m` away horizontally.
    Returns (lowest_ray_deg, height_on_post_m, slant_range_m).
    """
    alpha = math.radians(LOWEST_RAY_DEG)
    height = baseline_m * math.tan(alpha)
    slant = baseline_m / math.cos(alpha)
    return LOWEST_RAY_DEG, height, slant


# ── Calibration file paths ────────────────────────────────────────────────────

CALIB_PATH: Path = PROJECT_ROOT / "exports" / "stereo" / "calib.json"
OLD_CALIB_PATH: Path = PROJECT_ROOT / "exports" / "stereo" / "simple_focal.json"


def focal_axes(image_width: int, image_height: int) -> tuple[float, float]:
    """fx, fy in pixels from mounted FOVs and the actual frame size."""
    fx = (image_width / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    fy = (image_height / 2.0) / math.tan(math.radians(VFOV_DEG / 2.0))
    return fx, fy


def get_focal_px(image_width: int | None = None) -> tuple[float, str]:
    """
    Return (focal_px, source) — primarily the horizontal focal (fx).

    Priority: calib.json → simple_focal.json → spec from mounted HFOV + width.
    """
    w = image_width or NATIVE_WIDTH

    for path in (CALIB_PATH, OLD_CALIB_PATH):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            f = float(data["focal_px"])
            if f <= 0:
                continue
            saved_w = int(data.get("image_width") or NATIVE_WIDTH)
            if saved_w > 0 and saved_w != w:
                f = f * w / saved_w
                src = f"saved in {path.name}, scaled {saved_w}→{w} px"
            else:
                src = f"saved in {path.name}"
            return f, src
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue

    f = (w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    return f, f"spec-derived ({HFOV_DEG:.1f}° image-HFOV at {w} px wide)"


def save_calib(
    focal_px: float,
    baseline_m: float = BASELINE_M,
    h_angle_deg: float = H_ANGLE_DEG,
    v_angle_deg: float = V_ANGLE_DEG,
    image_width: int = NATIVE_WIDTH,
    image_height: int = NATIVE_HEIGHT,
    calibrated_at_height_m: float | None = None,
) -> None:
    """Write calibration to calib.json."""
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ray, h_post, slant = lowest_visible_on_far_post(baseline_m)
    payload: dict = {
        "focal_px": round(focal_px, 4),
        "baseline_m": baseline_m,
        "h_angle_deg": h_angle_deg,
        "v_angle_deg": v_angle_deg,
        "image_width": image_width,
        "image_height": image_height,
        "hfov_deg": HFOV_DEG,
        "vfov_deg": VFOV_DEG,
        "cameras_sideways": CAMERAS_SIDEWAYS,
        "datasheet_hfov_deg": DATASHEET_HFOV_DEG,
        "datasheet_vfov_deg": DATASHEET_VFOV_DEG,
        "lowest_ray_deg": round(ray, 2),
        "lowest_far_post_height_m": round(h_post, 3),
        "lowest_far_post_slant_m": round(slant, 3),
        "setup": "corner_mount_across_goal",
        "camera": "B0332 + LN013 NOIR (sideways)",
        "note": (
            "Cameras mounted sideways: image VFOV=70°, HFOV≈47.3°. "
            "Edit capture/stereo_config.py for rig geometry."
        ),
    }
    if calibrated_at_height_m is not None:
        payload["calibrated_at_height_m"] = calibrated_at_height_m
    CALIB_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {CALIB_PATH}")
    print(json.dumps(payload, indent=2))


def print_summary(focal_px: float, focal_src: str) -> None:
    """Print the active stereo rig config — called at startup of each script."""
    ray, h_post, slant = lowest_visible_on_far_post()
    print(
        f"\n{'─'*60}\n"
        f"  Hawkeye stereo rig  (cameras SIDEWAYS)\n"
        f"{'─'*60}\n"
        f"  Datasheet : {DATASHEET_HFOV_DEG}° / {DATASHEET_VFOV_DEG}° at "
        f"{DATASHEET_WIDTH}×{DATASHEET_HEIGHT}\n"
        f"  In image  : HFOV {HFOV_DEG}°  VFOV {VFOV_DEG}°  "
        f"(70° is UP–DOWN)\n"
        f"  Native res: {NATIVE_WIDTH}×{NATIVE_HEIGHT} px  (portrait after rotate)\n"
        f"  Baseline  : {BASELINE_M:.3f} m\n"
        f"  H-angle   : {H_ANGLE_DEG:.1f}°  (across goal → far post)\n"
        f"  V-angle   : {V_ANGLE_DEG:.1f}°  (up from horizontal)\n"
        f"  Lowest ray: {ray:.1f}° above horizontal\n"
        f"  Far post  : lowest visible ≈ {h_post:.2f} m up the post  "
        f"(slant {slant:.2f} m)\n"
        f"  Focal fx  : {focal_px:.1f}  ← {focal_src}\n"
        f"  Spec fx/fy: {FOCAL_FX_SPEC:.1f} / {FOCAL_FY_SPEC:.1f}\n"
        f"  Posts zone: X ∈ [{-BASELINE_M/2:.3f}, {BASELINE_M/2:.3f}] m\n"
        f"{'─'*60}\n"
    )
