"""Constant-velocity Kalman filter for ball centre prediction between frames."""

from __future__ import annotations

import math

import cv2
import numpy as np


class BallKalman:
    """
    Predict ball (x, y) from prior positions.

    Used with a search window: prefer YOLO boxes near the predicted point.
    """

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float = 4.0,
    ) -> None:
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._kf = self._new_filter()
        self.initialized = False
        self._speed_px = 0.0

    def _new_filter(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self._process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def reset(self) -> None:
        self.initialized = False
        self._speed_px = 0.0
        self._kf = self._new_filter()

    @property
    def speed_px(self) -> float:
        return self._speed_px

    def predict(self) -> tuple[float, float] | None:
        if not self.initialized:
            return None
        state = self._kf.predict()
        # Advance internal state so predictions keep moving on missed frames.
        # OpenCV only updates statePost in correct(), not predict().
        self._kf.statePost = state
        vx = float(state[2])
        vy = float(state[3])
        self._speed_px = math.hypot(vx, vy)
        return float(state[0]), float(state[1])

    def update(self, x: float, y: float) -> None:
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self._kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self._speed_px = 0.0
            return
        self._kf.correct(measurement)
        state = self._kf.statePost
        self._speed_px = math.hypot(float(state[2]), float(state[3]))

    def search_radius(
        self,
        base_radius: float = 120.0,
        max_radius: float = 400.0,
        speed_scale: float = 2.5,
    ) -> float:
        """Grow the search window when the ball is moving quickly."""
        if not self.initialized:
            return base_radius
        return min(max_radius, base_radius + self._speed_px * speed_scale)
