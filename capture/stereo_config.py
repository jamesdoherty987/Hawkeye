"""
stereo_config.py — Central configuration for the Hawkeye stereo rig.

Edit this file whenever the physical rig changes (baseline, angles, lens).
All three stereo scripts (stereo_record, stereo_calibrate, test_stereo_distance)
read their defaults from here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Camera: B0332  ·  Lens: LN013 NOIR
  HFOV 70°  |  VFOV ≈47.3°  |  Native 1280×800
  Focal-length tolerance: ±5 %  |  Distortion: 1.5 %
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rig geometry
────────────
  The two cameras are mounted on a rigid bar on the goalposts, facing inward
  and upward so their fields of view overlap above the goal area.

    Baseline : 128 cm  (tape-measured centre-to-centre)
    H-angle  : ~25°    (each camera rotated inward from straight-ahead)
    V-angle  : ~25°    (each camera tilted upward from horizontal)

  If you remount or re-angle the cameras, update BASELINE_M / H_ANGLE_DEG /
  V_ANGLE_DEG below and re-run stereo_calibrate.py to verify focal.

World coordinate frame used throughout (for reference)
───────────────────────────────────────────────────────
  Origin : midpoint between the two cameras
  X      : positive toward the RIGHT camera  (width of goal)
  Y      : positive UPWARD
  Z      : positive into the field (away from the goal)

  Ball between the posts ⟺  X ∈ [−baseline/2, +baseline/2]
  Ball above crossbar    ⟺  Y > crossbar_height (2.5 m for GAA)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Physical rig ─────────────────────────────────────────────────────────────

BASELINE_M: float = 1.28       # centre-to-centre camera separation (metres)
H_ANGLE_DEG: float = 0.0       # horizontal inward angle of EACH camera (degrees) — cameras face straight ahead, no toe-in
V_ANGLE_DEG: float = 55.0      # vertical upward angle of EACH camera (degrees)

# ── Camera spec: B0332 + LN013 NOIR lens ─────────────────────────────────────

HFOV_DEG: float = 70.0         # horizontal FOV at native resolution
VFOV_DEG: float = 47.3         # vertical FOV at native resolution
NATIVE_WIDTH: int = 1280       # native output width (pixels)
NATIVE_HEIGHT: int = 800       # native output height (pixels)
FOCAL_TOLERANCE_PCT: float = 5.0    # lens focal-length tolerance ±%
DISTORTION_PCT: float = 1.5         # barrel/radial distortion %

# ── Derived focal length from spec (no calibration object needed) ─────────────
# Formula: focal_px = (image_width / 2) / tan(HFOV / 2)
#          At 1280 px wide with 70° HFOV → ≈ 914 px
FOCAL_PX_SPEC: float = (NATIVE_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))

# ── Calibration file paths ────────────────────────────────────────────────────

CALIB_PATH: Path = PROJECT_ROOT / "exports" / "stereo" / "calib.json"
OLD_CALIB_PATH: Path = PROJECT_ROOT / "exports" / "stereo" / "simple_focal.json"


# ── Helpers used by all three stereo scripts ──────────────────────────────────

def get_focal_px(image_width: int | None = None) -> tuple[float, str]:
    """
    Return (focal_px, source_description) for the given image width.

    Priority:
      1. calib.json  — saved after C-key calibration (most accurate)
      2. simple_focal.json — older parallel-stereo calib (still valid for focal)
      3. Spec formula from HFOV_DEG + NATIVE_WIDTH  (good to ±5 %)

    focal_px is scaled proportionally when the saved calibration used a
    different image width than the one requested.
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

    # Fall back to spec-derived value
    f = (w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    return f, f"spec-derived (70° HFOV at {w} px wide)"


def save_calib(
    focal_px: float,
    baseline_m: float = BASELINE_M,
    h_angle_deg: float = H_ANGLE_DEG,
    v_angle_deg: float = V_ANGLE_DEG,
    image_width: int = NATIVE_WIDTH,
    image_height: int = NATIVE_HEIGHT,
    calibrated_at_height_m: float | None = None,
) -> None:
    """Write calibration to calib.json. Called by stereo_calibrate.py on C or Q."""
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "focal_px": round(focal_px, 4),
        "baseline_m": baseline_m,
        "h_angle_deg": h_angle_deg,
        "v_angle_deg": v_angle_deg,
        "image_width": image_width,
        "image_height": image_height,
        "hfov_deg": HFOV_DEG,
        "vfov_deg": VFOV_DEG,
        "setup": "converging",
        "camera": "B0332 + LN013 NOIR",
        "note": (
            "Edit capture/stereo_config.py for rig geometry. "
            "Re-run stereo_calibrate.py (C key) if lens or resolution changes."
        ),
    }
    if calibrated_at_height_m is not None:
        payload["calibrated_at_height_m"] = calibrated_at_height_m
    CALIB_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {CALIB_PATH}")
    print(json.dumps(payload, indent=2))


def print_summary(focal_px: float, focal_src: str) -> None:
    """Print the active stereo rig config — called at startup of each script."""
    print(
        f"\n{'─'*60}\n"
        f"  Hawkeye stereo rig\n"
        f"{'─'*60}\n"
        f"  Camera    : B0332 + LN013 NOIR  ({HFOV_DEG}° HFOV, {VFOV_DEG}° VFOV)\n"
        f"  Native res: {NATIVE_WIDTH}×{NATIVE_HEIGHT} px\n"
        f"  Baseline  : {BASELINE_M:.3f} m\n"
        f"  H-angle   : {H_ANGLE_DEG:.1f}°  (each camera inward)\n"
        f"  V-angle   : {V_ANGLE_DEG:.1f}°  (each camera upward)\n"
        f"  Focal px  : {focal_px:.1f}  ← {focal_src}\n"
        f"  Spec focal: {FOCAL_PX_SPEC:.1f}  (from 70° HFOV at {NATIVE_WIDTH} px)\n"
        f"  Posts zone: X ∈ [{-BASELINE_M/2:.3f}, {BASELINE_M/2:.3f}] m\n"
        f"{'─'*60}\n"
    )
