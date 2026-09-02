"""
Motion detection for sky-facing cameras (static background, fast ball).

Uses a hybrid of:
  - Running-average background model (learns "normal sky")
  - Frame differencing (catches fast movers MOG2 often misses)

OpenCV-only — no extra dependencies. Based on OpenCV bg-subtraction guidance:
fast objects are better served by frame differencing than slow-adapting MOG2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionBlob:
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    area: float
    compactness: float  # contour area / bbox area (1.0 = solid fill)


@dataclass
class MotionResult:
    blobs: list[MotionBlob] = field(default_factory=list)
    ready: bool = False
    mask: np.ndarray | None = None


class SkyMotionDetector:
    """
    Detect moving regions against a learned sky background.

    Call process(frame) each frame. Background stabilises over warmup_frames.
    """

    def __init__(
        self,
        warmup_frames: int = 25,
        warmup_learning_rate: float = 0.08,
        steady_learning_rate: float = 0.01,
        diff_threshold: int = 22,
        blur_ksize: int = 5,
        min_blob_area: float = 25.0,
        max_blob_area: float = 12000.0,
        min_compactness: float = 0.20,
        max_aspect_ratio: float = 4.0,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.warmup_learning_rate = warmup_learning_rate
        self.steady_learning_rate = steady_learning_rate
        self.diff_threshold = diff_threshold
        self.blur_ksize = blur_ksize | 1  # must be odd
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.min_compactness = min_compactness
        self.max_aspect_ratio = max_aspect_ratio

        self._background: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._frame_count = 0
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @property
    def ready(self) -> bool:
        return self._frame_count >= self.warmup_frames

    def reset(self) -> None:
        self._background = None
        self._prev_gray = None
        self._frame_count = 0

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

    def _update_background(self, gray: np.ndarray) -> None:
        gray_f = gray.astype(np.float32)
        if self._background is None:
            self._background = gray_f.copy()
            return

        alpha = (
            self.warmup_learning_rate
            if self._frame_count < self.warmup_frames
            else self.steady_learning_rate
        )
        cv2.accumulateWeighted(gray, self._background, alpha)

    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        bg = self._background.astype(np.uint8)
        bg_diff = cv2.absdiff(gray, bg)

        frame_diff = np.zeros_like(gray)
        if self._prev_gray is not None:
            frame_diff = cv2.absdiff(gray, self._prev_gray)

        combined = cv2.max(bg_diff, frame_diff)
        _, mask = cv2.threshold(
            combined, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        return mask

    def _extract_blobs(self, mask: np.ndarray) -> list[MotionBlob]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[MotionBlob] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 2 or h < 2:
                continue

            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect_ratio:
                continue

            bbox_area = float(w * h)
            compactness = area / bbox_area if bbox_area > 0 else 0.0
            if compactness < self.min_compactness:
                continue

            blobs.append(
                MotionBlob(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    cx=x + w / 2.0,
                    cy=y + h / 2.0,
                    area=area,
                    compactness=compactness,
                )
            )

        blobs.sort(key=lambda b: b.area)
        return blobs

    def process(self, frame: np.ndarray, return_mask: bool = False) -> MotionResult:
        gray = self._preprocess(frame)
        self._update_background(gray)
        mask = self._motion_mask(gray)
        self._prev_gray = gray
        self._frame_count += 1

        ready = self._frame_count >= self.warmup_frames
        blobs = self._extract_blobs(mask) if ready else []

        return MotionResult(
            blobs=blobs,
            ready=ready,
            mask=mask if return_mask else None,
        )


def box_overlaps_motion(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    blobs: list[MotionBlob],
) -> float:
    """
    Best overlap score in [0, 1] between a box and any motion blob.

    Uses max of centre-inside-blob and IoU.
    """
    if not blobs:
        return 0.0

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_area = max((x2 - x1) * (y2 - y1), 1.0)
    best = 0.0

    for blob in blobs:
        if blob.x1 <= cx <= blob.x2 and blob.y1 <= cy <= blob.y2:
            return 1.0

        ix1 = max(x1, blob.x1)
        iy1 = max(y1, blob.y1)
        ix2 = min(x2, blob.x2)
        iy2 = min(y2, blob.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = box_area + blob.area - inter
        iou = inter / max(union, 1.0)
        best = max(best, iou)

    return best


def nearest_motion_blob(
    blobs: list[MotionBlob],
    point: tuple[float, float],
    max_distance: float,
) -> MotionBlob | None:
    if not blobs:
        return None

    px, py = point
    best: MotionBlob | None = None
    best_dist = max_distance

    for blob in blobs:
        dist = math.hypot(blob.cx - px, blob.cy - py)
        if dist <= best_dist:
            best_dist = dist
            best = blob

    return best


def draw_motion_overlay(frame: np.ndarray, motion: MotionResult) -> None:
    """Draw motion blobs (green) and optional mask tint for debugging."""
    for blob in motion.blobs:
        cv2.rectangle(
            frame,
            (blob.x1, blob.y1),
            (blob.x2, blob.y2),
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (int(blob.cx), int(blob.cy)), 3, (0, 220, 0), -1)
