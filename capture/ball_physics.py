"""
Parabolic ball-flight model in image space.

A fixed camera sees a thrown ball as a smooth curved path (roughly quadratic in
time for each axis). We fit recent positions, predict the next point, and reject
detections that teleport or move at impossible speed — the main cause of sharp
trail glitches.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# Generous limits for a ball crossing a 720p–1080p sky frame.
MAX_SPEED_PX_S = 4000.0
MAX_ACCEL_PX_S2 = 30000.0
MIN_POINTS_QUADRATIC = 6
MIN_TIME_SPAN_S = 1.0 / 120.0


class BallFlightModel:
    """Fit short arcs to recent (t, x, y) samples and gate new measurements."""

    def __init__(self, history: int = 24) -> None:
        self._points: deque[tuple[float, float, float]] = deque(maxlen=history)

    @property
    def point_count(self) -> int:
        return len(self._points)

    def reset(self) -> None:
        self._points.clear()

    def add(self, t: float, x: float, y: float) -> None:
        self._points.append((t, x, y))

    def can_predict(self) -> bool:
        return len(self._points) >= 2

    def _speed_estimate(self) -> float:
        if len(self._points) < 2:
            return 0.0
        t0, x0, y0 = self._points[-2]
        t1, x1, y1 = self._points[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    @staticmethod
    def _dedupe_times(
        tn: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop duplicate timestamps (keeps last sample at each t)."""
        if len(tn) <= 1:
            return tn, xs, ys
        order = np.argsort(tn)
        tn, xs, ys = tn[order], xs[order], ys[order]
        keep = np.ones(len(tn), dtype=bool)
        keep[:-1] = np.diff(tn) > 1e-9
        return tn[keep], xs[keep], ys[keep]

    @staticmethod
    def _fit_axis(tn: np.ndarray, values: np.ndarray, max_degree: int) -> np.ndarray:
        """Robust polyfit with degree fallback for sparse/degenerate samples."""
        n = len(tn)
        if n == 0:
            return np.array([0.0])
        if n == 1 or float(np.std(tn)) < MIN_TIME_SPAN_S:
            return np.array([float(values[-1])])

        cap = min(max_degree, n - 1)
        for degree in range(cap, 0, -1):
            try:
                coeff = np.polyfit(tn, values, degree)
                if np.all(np.isfinite(coeff)):
                    return coeff
            except (np.linalg.LinAlgError, ValueError):
                continue

        dt = float(tn[-1] - tn[-2])
        if abs(dt) > MIN_TIME_SPAN_S:
            slope = float((values[-1] - values[-2]) / dt)
            return np.array([slope, float(values[-1])])
        return np.array([float(values[-1])])

    def _fit_coeffs(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        n = len(self._points)
        if n < 2:
            return None

        ts = np.array([p[0] for p in self._points], dtype=np.float64)
        xs = np.array([p[1] for p in self._points], dtype=np.float64)
        ys = np.array([p[2] for p in self._points], dtype=np.float64)

        t_ref = float(ts[-1])
        tn, xs, ys = self._dedupe_times(ts - t_ref, xs, ys)
        if len(tn) < 2:
            return None

        max_degree = 2 if len(tn) >= MIN_POINTS_QUADRATIC else 1
        try:
            px = self._fit_axis(tn, xs, max_degree)
            py = self._fit_axis(tn, ys, max_degree)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return t_ref, px, py

    def predict(self, t: float) -> tuple[float, float] | None:
        try:
            fit = self._fit_coeffs()
        except (np.linalg.LinAlgError, ValueError):
            fit = None
        if fit is None:
            if self._points:
                return self._points[-1][1], self._points[-1][2]
            return None

        t_ref, px, py = fit
        dt = t - t_ref
        x = float(np.polyval(px, dt))
        y = float(np.polyval(py, dt))
        if not (math.isfinite(x) and math.isfinite(y)):
            return self._points[-1][1], self._points[-1][2]
        return x, y

    def velocity_at(self, t: float) -> tuple[float, float] | None:
        try:
            fit = self._fit_coeffs()
        except (np.linalg.LinAlgError, ValueError):
            return None
        if fit is None:
            return None
        t_ref, px, py = fit
        dt = t - t_ref
        degree = len(px) - 1
        if degree <= 0:
            return 0.0, 0.0
        if degree == 1:
            return float(px[0]), float(py[0])
        vx = float(2.0 * px[0] * dt + px[1])
        vy = float(2.0 * py[0] * dt + py[1])
        if not (math.isfinite(vx) and math.isfinite(vy)):
            return None
        return vx, vy

    def gate(
        self,
        t: float,
        x: float,
        y: float,
        fps: float,
    ) -> tuple[bool, str]:
        """Return (accepted, reason). Reject teleports and non-ball motion."""
        if not self._points:
            return True, "bootstrap"

        last_t, last_x, last_y = self._points[-1]
        dt = t - last_t
        if dt <= 1e-6:
            return True, "same_time"

        dist_last = math.hypot(x - last_x, y - last_y)
        inst_speed = dist_last / dt
        if inst_speed > MAX_SPEED_PX_S:
            return False, f"speed {inst_speed:.0f}px/s"

        try:
            pred = self.predict(t)
        except (np.linalg.LinAlgError, ValueError):
            return True, "no_model"
        if pred is None:
            return True, "no_model"

        dist_pred = math.hypot(x - pred[0], y - pred[1])
        speed_est = self._speed_estimate()
        frame_dt = max(1.0 / max(fps, 1.0), 1.0 / 120.0)
        max_jump = max(90.0, speed_est * frame_dt * 4.5, inst_speed * frame_dt * 2.0)

        if dist_pred > max_jump:
            return False, f"off_arc {dist_pred:.0f}px"

        vel_now = self.velocity_at(t)
        if vel_now is not None and len(self._points) >= 2:
            vel_last = self.velocity_at(last_t)
            if vel_last is not None:
                ax = (vel_now[0] - vel_last[0]) / max(dt, 1e-6)
                ay = (vel_now[1] - vel_last[1]) / max(dt, 1e-6)
                accel = math.hypot(ax, ay)
                if accel > MAX_ACCEL_PX_S2:
                    return False, f"accel {accel:.0f}px/s2"

        return True, "ok"

    def gate_relaxed(
        self,
        t: float,
        x: float,
        y: float,
        fps: float,
    ) -> tuple[bool, str]:
        """Looser gating for offline path reconstruction (speed cap only)."""
        if not self._points:
            return True, "bootstrap"

        last_t, last_x, last_y = self._points[-1]
        dt = t - last_t
        if dt <= 1e-6:
            return True, "same_time"

        dist = math.hypot(x - last_x, y - last_y)
        inst_speed = dist / dt
        if inst_speed > MAX_SPEED_PX_S * 1.5:
            return False, f"speed {inst_speed:.0f}px/s"

        frame_dt = max(1.0 / max(fps, 1.0), 1.0 / 120.0)
        speed_est = self._speed_estimate()
        max_jump = max(120.0, speed_est * frame_dt * 6.0)
        if dist > max_jump:
            return False, f"jump {dist:.0f}px"

        return True, "ok"

    def search_radius(
        self,
        base_radius: float = 120.0,
        max_radius: float = 400.0,
        speed_scale: float = 2.5,
    ) -> float:
        return min(max_radius, base_radius + self._speed_estimate() * speed_scale)

    def arc_ahead(
        self,
        t_now: float,
        horizon_s: float = 0.35,
        steps: int = 10,
    ) -> list[tuple[int, int]]:
        """Sample points along the fitted arc for overlay drawing."""
        if not self.can_predict():
            return []
        pts: list[tuple[int, int]] = []
        for i in range(1, steps + 1):
            t = t_now + horizon_s * i / steps
            try:
                p = self.predict(t)
            except (np.linalg.LinAlgError, ValueError):
                break
            if p is not None and math.isfinite(p[0]) and math.isfinite(p[1]):
                pts.append((int(p[0]), int(p[1])))
        return pts
