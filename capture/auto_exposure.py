"""
Software auto-exposure for sky-facing USB cameras.

Meters a horizontal band through the middle of the frame (where the ball
crosses) instead of the whole image, then nudges exposure/gain toward a
target brightness. Locks after settle so live detect/track stay stable.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

# Sky-facing setup: meter the middle band where the ball crosses the frame.
TARGET_BRIGHTNESS = 130.0  # 0–255; slightly bright keeps ball contrast on sky
BRIGHTNESS_TOLERANCE = 12.0
ROI_Y_FRAC = (0.35, 0.65)
ROI_X_FRAC = (0.10, 0.90)

# Typical Windows/Mac UVC exposure range is about -13 (dark) to -1 (bright).
MIN_EXPOSURE = -13.0
MAX_EXPOSURE = -1.0
DEFAULT_EXPOSURE = -8.0
EXPOSURE_STEP = 1.0
GAIN_MIN = 0.0
GAIN_MAX = 64.0
DEFAULT_GAIN = 0.0
GAIN_STEP = 2.0
AE_EVERY_N_FRAMES = 5
AE_SETTLE_LOOPS = 25
LOCK_AFTER_SETTLE = True


def try_set(cap: cv2.VideoCapture, prop: int, value: float) -> bool:
    try:
        return bool(cap.set(prop, value))
    except Exception:
        return False


def force_manual_exposure_mode(cap: cv2.VideoCapture) -> None:
    """Turn off the camera's own AE so our software loop can drive exposure."""
    for value in (0.25, 1.0, 0.0):
        if try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, value):
            break
    try_set(cap, cv2.CAP_PROP_AUTO_WB, 1)


def frame_brightness(
    frame,
    roi_y_frac: tuple[float, float] = ROI_Y_FRAC,
    roi_x_frac: tuple[float, float] = ROI_X_FRAC,
) -> float:
    """Mean brightness of the ball-path ROI (0–255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0 = int(h * roi_y_frac[0])
    y1 = int(h * roi_y_frac[1])
    x0 = int(w * roi_x_frac[0])
    x1 = int(w * roi_x_frac[1])
    return float(np.mean(gray[y0:y1, x0:x1]))


def read_frame(cap: cv2.VideoCapture, retries: int = 5) -> tuple[bool, object | None]:
    for _ in range(max(retries, 1)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.02)
    return False, None


class SoftwareAutoExposure:
    """
    Nudge camera exposure/gain toward a sensible brightness for sky views.

    Hardware auto-exposure often fails when most of the frame is bright sky.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.exposure = DEFAULT_EXPOSURE
        self.gain = DEFAULT_GAIN
        self.frame_count = 0
        self.last_brightness = 0.0
        self.locked = False
        force_manual_exposure_mode(cap)
        try_set(cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
        try_set(cap, cv2.CAP_PROP_GAIN, self.gain)

    def status_text(self) -> str:
        lock = "locked" if self.locked else "adjusting"
        return (
            f"AutoExp ({lock}) brightness={self.last_brightness:.0f} "
            f"exp={self.exposure:.0f} gain={self.gain:.0f}"
        )

    def lock(self) -> None:
        self.locked = True

    def unlock(self) -> None:
        self.locked = False

    def update(self, frame, force: bool = False) -> None:
        if self.locked and not force:
            return

        self.frame_count += 1
        if not force and self.frame_count % AE_EVERY_N_FRAMES != 0:
            return

        brightness = frame_brightness(frame)
        self.last_brightness = brightness
        error = brightness - TARGET_BRIGHTNESS

        if abs(error) <= BRIGHTNESS_TOLERANCE:
            return

        # Too bright → lower exposure first, then gain.
        # Too dark → raise exposure first, then gain.
        if error > 0:
            if self.exposure > MIN_EXPOSURE:
                self.exposure = max(MIN_EXPOSURE, self.exposure - EXPOSURE_STEP)
                try_set(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
            elif self.gain > GAIN_MIN:
                self.gain = max(GAIN_MIN, self.gain - GAIN_STEP)
                try_set(self.cap, cv2.CAP_PROP_GAIN, self.gain)
        else:
            if self.exposure < MAX_EXPOSURE:
                self.exposure = min(MAX_EXPOSURE, self.exposure + EXPOSURE_STEP)
                try_set(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
            elif self.gain < GAIN_MAX:
                self.gain = min(GAIN_MAX, self.gain + GAIN_STEP)
                try_set(self.cap, cv2.CAP_PROP_GAIN, self.gain)

    def settle(self, lock: bool = LOCK_AFTER_SETTLE) -> None:
        """Run a short loop so sky brightness is corrected before tracking."""
        print("Auto-adjusting exposure for sky view (ball-path band)...")
        for _ in range(AE_SETTLE_LOOPS):
            ok, frame = read_frame(self.cap)
            if not ok or frame is None:
                continue
            self.update(frame, force=True)
        if lock:
            self.lock()
        elif self.last_brightness == 0.0:
            ok, frame = read_frame(self.cap)
            if ok and frame is not None:
                self.last_brightness = frame_brightness(frame)
        print(f"Exposure ready: {self.status_text()}")
