"""
Software auto-exposure for sky-facing USB cameras.

Meters a horizontal band through the middle of the frame (where the ball
crosses) instead of the whole image. Uses camera exposure/gain when the
driver supports it, and always applies software tone mapping on top so
direct sun / sky views do not blow out.
"""

from __future__ import annotations

import platform
import time

import cv2
import numpy as np

# Sky-facing setup: meter the middle band where the ball crosses the frame.
TARGET_BRIGHTNESS = 120.0  # 0–255; slightly below mid keeps ball contrast on sky
BRIGHTNESS_TOLERANCE = 10.0
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
AE_EVERY_N_FRAMES = 2
AE_SETTLE_LOOPS = 100
AE_SETTLE_AFTER_CHANGE_S = 0.05
LOCK_AFTER_SETTLE = False
SOFTWARE_COMP_MIN = 0.02
SOFTWARE_COMP_MAX = 1.0
OUTDOOR_RAW_THRESHOLD = 200.0


def try_set(cap: cv2.VideoCapture, prop: int, value: float) -> bool:
    try:
        return bool(cap.set(prop, value))
    except Exception:
        return False


def try_get(cap: cv2.VideoCapture, prop: int, default: float = 0.0) -> float:
    try:
        value = cap.get(prop)
        if value is None or value < 0:
            return default
        return float(value)
    except Exception:
        return default


def manual_auto_exposure_values() -> tuple[float, ...]:
    """Values that mean 'manual exposure' vary by OS/driver."""
    system = platform.system()
    if system == "Linux":
        return (1.0, 0.25, 0.0)
    if system == "Darwin":
        return (0.0, 0.25)
    # Windows MSMF/DirectShow: 0.25 = manual, 0.75/1.0 = auto — do not use 1.0 here.
    return (0.25, 0.0)


def force_manual_exposure_mode(cap: cv2.VideoCapture) -> None:
    """Turn off the camera's own AE so our software loop can drive exposure."""
    for value in manual_auto_exposure_values():
        if try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, value):
            break
    try_set(cap, cv2.CAP_PROP_AUTO_WB, 1)


def frame_brightness(
    frame,
    roi_y_frac: tuple[float, float] = ROI_Y_FRAC,
    roi_x_frac: tuple[float, float] = ROI_X_FRAC,
) -> float:
    """Median brightness of the ball-path ROI (0–255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0 = int(h * roi_y_frac[0])
    y1 = int(h * roi_y_frac[1])
    x0 = int(w * roi_x_frac[0])
    x1 = int(w * roi_x_frac[1])
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return float(np.mean(gray))
    return float(np.median(roi))


def read_frame(cap: cv2.VideoCapture, retries: int = 5) -> tuple[bool, object | None]:
    for _ in range(max(retries, 1)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.02)
    return False, None


def exposure_step_for_error(error: float) -> float:
    magnitude = abs(error)
    if magnitude > 100:
        return 4.0
    if magnitude > 50:
        return 2.0
    return EXPOSURE_STEP


def gain_step_for_error(error: float) -> float:
    magnitude = abs(error)
    if magnitude > 100:
        return 10.0
    if magnitude > 50:
        return 5.0
    return GAIN_STEP


def software_blend_step(raw_brightness: float, output_error: float) -> float:
    """How fast to move software compensation toward the target."""
    if abs(output_error) > 100:
        return 0.92
    if abs(output_error) > 50:
        return 0.80
    if raw_brightness >= OUTDOOR_RAW_THRESHOLD:
        return 0.70
    return 0.50


class SoftwareAutoExposure:
    """
    Hybrid auto-exposure for sky-facing cameras.

    Hardware exposure/gain is used when the driver responds, but software
    tone mapping always runs as a second stage so outdoor sky views do not
    stay blown out when the sensor minimum is still too bright.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.min_exposure = MIN_EXPOSURE
        self.max_exposure = MAX_EXPOSURE
        self.exposure = DEFAULT_EXPOSURE
        self.gain = DEFAULT_GAIN
        self.camera_brightness = 128.0
        self.frame_count = 0
        self.last_raw_brightness = 0.0
        self.last_brightness = 0.0
        self.locked = False
        self.hardware_controls = True
        self.software_compensation = SOFTWARE_COMP_MAX
        self._last_hw_adjust_brightness: float | None = None
        self._hw_stuck_count = 0

        force_manual_exposure_mode(cap)
        self._read_camera_exposure_limits()
        self.camera_brightness = try_get(cap, cv2.CAP_PROP_BRIGHTNESS, 128.0)
        try_set(cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
        try_set(cap, cv2.CAP_PROP_GAIN, self.gain)
        self.hardware_controls = self._probe_hardware_exposure()
        self._push_hardware_to_minimum_if_bright()

    def _read_camera_exposure_limits(self) -> None:
        current = try_get(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
        if current != 0.0:
            self.exposure = current

    def _push_hardware_to_minimum_if_bright(self) -> None:
        """Start outdoor sessions with the shortest exposure the driver allows."""
        ok, frame = read_frame(self.cap, retries=8)
        if not ok or frame is None:
            return

        raw = frame_brightness(frame)
        self.last_raw_brightness = raw
        if raw < OUTDOOR_RAW_THRESHOLD:
            return

        print(f"Bright scene detected (raw={raw:.0f}); pushing camera to minimum exposure.")
        self.exposure = self.min_exposure
        self.gain = GAIN_MIN
        try_set(self.cap, cv2.CAP_PROP_GAIN, self.gain)
        try_set(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure)
        self._lower_camera_brightness()
        time.sleep(AE_SETTLE_AFTER_CHANGE_S)

        ok, frame = read_frame(self.cap, retries=8)
        if ok and frame is not None:
            raw = frame_brightness(frame)
            self.last_raw_brightness = raw
            if raw > TARGET_BRIGHTNESS + BRIGHTNESS_TOLERANCE:
                desired = TARGET_BRIGHTNESS / raw
                self.software_compensation = float(
                    np.clip(desired, SOFTWARE_COMP_MIN, SOFTWARE_COMP_MAX)
                )

    def _lower_camera_brightness(self) -> None:
        """Some UVC drivers expose a brightness slider separate from exposure."""
        if self.camera_brightness <= 0:
            return
        self.camera_brightness = 0.0
        try_set(self.cap, cv2.CAP_PROP_BRIGHTNESS, self.camera_brightness)

    def _probe_hardware_exposure(self) -> bool:
        """Return False when exposure property changes do not affect brightness."""
        original_exp = self.exposure
        original_gain = self.gain

        ok, frame = read_frame(self.cap, retries=8)
        if not ok or frame is None:
            return True

        bright_before = frame_brightness(frame)

        dark_exp = self.min_exposure
        bright_exp = self.max_exposure
        try_set(self.cap, cv2.CAP_PROP_GAIN, GAIN_MIN)
        try_set(self.cap, cv2.CAP_PROP_EXPOSURE, dark_exp)
        time.sleep(AE_SETTLE_AFTER_CHANGE_S)
        ok, dark_frame = read_frame(self.cap, retries=8)
        bright_at_dark = (
            frame_brightness(dark_frame) if ok and dark_frame is not None else bright_before
        )

        try_set(self.cap, cv2.CAP_PROP_EXPOSURE, bright_exp)
        time.sleep(AE_SETTLE_AFTER_CHANGE_S)
        ok, bright_frame = read_frame(self.cap, retries=8)
        bright_at_bright = (
            frame_brightness(bright_frame) if ok and bright_frame is not None else bright_before
        )

        try_set(self.cap, cv2.CAP_PROP_EXPOSURE, original_exp)
        try_set(self.cap, cv2.CAP_PROP_GAIN, original_gain)
        time.sleep(AE_SETTLE_AFTER_CHANGE_S)

        delta = abs(bright_at_bright - bright_at_dark)
        if delta < 8.0:
            print(
                "Camera exposure property has little effect; relying on software "
                "tone mapping for bright scenes."
            )
            return False

        if bright_at_dark > bright_at_bright:
            self.min_exposure, self.max_exposure = self.max_exposure, self.min_exposure
            print("Inverted camera exposure range detected.")

        if bright_at_dark > OUTDOOR_RAW_THRESHOLD:
            print(
                "Minimum camera exposure is still very bright outdoors; software "
                "tone mapping will be used."
            )

        return True

    def status_text(self) -> str:
        lock = "locked" if self.locked else "adjusting"
        mode = "hybrid" if self.hardware_controls else "sw"
        return (
            f"AutoExp ({lock}/{mode}) raw={self.last_raw_brightness:.0f} "
            f"out={self.last_brightness:.0f} exp={self.exposure:.0f} "
            f"gain={self.gain:.0f} comp={self.software_compensation:.2f}"
        )

    def lock(self) -> None:
        self.locked = True

    def unlock(self) -> None:
        self.locked = False

    def apply(self, frame):
        """Return frame with software tone mapping applied."""
        if self.software_compensation >= 0.999:
            return frame

        # Linear scale works even when the sensor has clipped to pure white.
        corrected = np.clip(
            frame.astype(np.float32) * self.software_compensation,
            0,
            255,
        )
        return corrected.astype(np.uint8)

    def process(self, frame, force: bool = False):
        """Update exposure from frame and return a corrected copy."""
        self.update(frame, force=force)
        return self.apply(frame)

    def _hardware_exhausted(self, error: float) -> bool:
        if error > 0:
            return self.exposure <= self.min_exposure and self.gain <= GAIN_MIN
        return self.exposure >= self.max_exposure and self.gain >= GAIN_MAX

    def _adjust_hardware(self, error: float, wait: bool = False) -> None:
        exp_step = exposure_step_for_error(error)
        gain_step = gain_step_for_error(error)

        if error > 0:
            if self.exposure > self.min_exposure:
                self.exposure = max(self.min_exposure, self.exposure - exp_step)
                if try_set(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure) and wait:
                    time.sleep(AE_SETTLE_AFTER_CHANGE_S)
            elif self.gain > GAIN_MIN:
                self.gain = max(GAIN_MIN, self.gain - gain_step)
                try_set(self.cap, cv2.CAP_PROP_GAIN, self.gain)
            elif self.camera_brightness > 0:
                self._lower_camera_brightness()
        else:
            if self.exposure < self.max_exposure:
                self.exposure = min(self.max_exposure, self.exposure + exp_step)
                if try_set(self.cap, cv2.CAP_PROP_EXPOSURE, self.exposure) and wait:
                    time.sleep(AE_SETTLE_AFTER_CHANGE_S)
            elif self.gain < GAIN_MAX:
                self.gain = min(GAIN_MAX, self.gain + gain_step)
                try_set(self.cap, cv2.CAP_PROP_GAIN, self.gain)

    def _adjust_software(self, raw_brightness: float, output_brightness: float) -> None:
        output_error = output_brightness - TARGET_BRIGHTNESS
        if abs(output_error) <= BRIGHTNESS_TOLERANCE:
            if (
                raw_brightness < TARGET_BRIGHTNESS - BRIGHTNESS_TOLERANCE
                and self.software_compensation < SOFTWARE_COMP_MAX
            ):
                self.software_compensation = float(
                    min(
                        SOFTWARE_COMP_MAX,
                        self.software_compensation + 0.08,
                    )
                )
            return

        if output_error > 0:
            desired = TARGET_BRIGHTNESS / max(output_brightness, 1.0)
            desired = float(np.clip(desired, SOFTWARE_COMP_MIN, SOFTWARE_COMP_MAX))
            step = software_blend_step(raw_brightness, output_error)
            self.software_compensation = float(
                self.software_compensation
                + (desired - self.software_compensation) * step
            )
        elif self.software_compensation < SOFTWARE_COMP_MAX:
            self.software_compensation = float(
                min(
                    SOFTWARE_COMP_MAX,
                    self.software_compensation + 0.06,
                )
            )

    def _track_hardware_effect(self, brightness: float, error: float) -> None:
        if not self.hardware_controls or abs(error) <= BRIGHTNESS_TOLERANCE:
            self._hw_stuck_count = 0
            return

        if self._last_hw_adjust_brightness is None:
            self._last_hw_adjust_brightness = brightness
            return

        if abs(brightness - self._last_hw_adjust_brightness) < 3.0:
            self._hw_stuck_count += 1
        else:
            self._hw_stuck_count = 0

        self._last_hw_adjust_brightness = brightness

        if self._hw_stuck_count >= 5:
            self.hardware_controls = False
            self._hw_stuck_count = 0
            print(
                "Hardware exposure stopped responding; software tone mapping "
                "will carry the full correction."
            )

    def update(self, frame, force: bool = False) -> None:
        if self.locked and not force:
            return

        raw_brightness = frame_brightness(frame)
        self.last_raw_brightness = raw_brightness

        if raw_brightness > TARGET_BRIGHTNESS + BRIGHTNESS_TOLERANCE:
            if self.software_compensation >= 0.95:
                instant = TARGET_BRIGHTNESS / raw_brightness
                self.software_compensation = float(
                    np.clip(instant, SOFTWARE_COMP_MIN, SOFTWARE_COMP_MAX)
                )

        output_brightness = raw_brightness * self.software_compensation
        self.last_brightness = output_brightness
        self._adjust_software(raw_brightness, output_brightness)
        self.last_brightness = raw_brightness * self.software_compensation

        raw_error = raw_brightness - TARGET_BRIGHTNESS
        if not self.hardware_controls or abs(raw_error) <= BRIGHTNESS_TOLERANCE:
            return

        self.frame_count += 1
        if not force and self.frame_count % AE_EVERY_N_FRAMES != 0:
            return

        self._adjust_hardware(raw_error, wait=force)
        if self._hardware_exhausted(raw_error):
            self.hardware_controls = False
        else:
            self._track_hardware_effect(raw_brightness, raw_error)

    def settle(self, lock: bool = LOCK_AFTER_SETTLE) -> None:
        """Run a loop so sky brightness is corrected before tracking."""
        print("Auto-adjusting exposure for sky view (ball-path band)...")
        stable_count = 0
        for _ in range(AE_SETTLE_LOOPS):
            ok, frame = read_frame(self.cap)
            if not ok or frame is None:
                continue
            self.process(frame, force=True)
            if abs(self.last_brightness - TARGET_BRIGHTNESS) <= BRIGHTNESS_TOLERANCE:
                stable_count += 1
                if stable_count >= 4:
                    break
            else:
                stable_count = 0

        if lock:
            self.lock()
        elif self.last_brightness == 0.0:
            ok, frame = read_frame(self.cap)
            if ok and frame is not None:
                self.last_brightness = frame_brightness(self.apply(frame))
        print(f"Exposure ready: {self.status_text()}")
