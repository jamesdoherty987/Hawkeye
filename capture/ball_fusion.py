"""
Fuse sky motion blobs with YOLO football detections into one pick per frame.

Pipeline:
  1. Motion finds moving regions (sky background model).
  2. YOLO full-frame track proposes football boxes.
  3. Keep YOLO hits that overlap motion (drops static false positives).
  4. For motion blobs without YOLO, run a tight crop re-check at lower conf.
  5. Kalman prediction breaks ties and enables brief motion coast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ultralytics import YOLO

from sky_motion import (
    MotionBlob,
    MotionResult,
    box_overlaps_motion,
    nearest_motion_blob,
)


MOTION_CROP_CONF = 0.22
MOTION_CROP_PAD_FRAC = 0.55
MOTION_CROP_MAX = 4
MOTION_GATE_OVERLAP = 0.10
MOTION_MATCH_OVERLAP = 0.15
HIGH_CONF_NO_MOTION = 0.72
KALMAN_DIST_PENALTY = 4.0
MOTION_OVERLAP_WEIGHT = 700.0


@dataclass(frozen=True)
class FusionPick:
    track_id: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    source: str  # yolo+motion | yolo+crop | motion+kalman | yolo


@dataclass
class YoloDet:
    track_id: int | None
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def parse_track_boxes(boxes, min_conf: float) -> list[YoloDet]:
    if boxes is None:
        return []

    dets: list[YoloDet] = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        tid = int(box.id[0]) if box.id is not None else None
        dets.append(YoloDet(tid, conf, x1, y1, x2, y2))
    return dets


def blob_padded_box(blob: MotionBlob, pad_scale: float = 0.5) -> tuple[float, float, float, float]:
    pad = max(4, int(math.sqrt(blob.area) * pad_scale))
    return (
        float(blob.x1 - pad),
        float(blob.y1 - pad),
        float(blob.x2 + pad),
        float(blob.y2 + pad),
    )


def merge_yolo_motion_boxes(
    yolo: YoloDet,
    blob: MotionBlob,
) -> tuple[float, float, float, float]:
    """Prefer YOLO box geometry; expand slightly with motion if YOLO is tiny."""
    x1, y1, x2, y2 = yolo.x1, yolo.y1, yolo.x2, yolo.y2
    yw, yh = x2 - x1, y2 - y1
    if yw < 8 or yh < 8:
        bx1, by1, bx2, by2 = blob_padded_box(blob, pad_scale=0.35)
        x1 = min(x1, bx1)
        y1 = min(y1, by1)
        x2 = max(x2, bx2)
        y2 = max(y2, by2)
    return x1, y1, x2, y2


def best_blob_for_point(blobs: list[MotionBlob], cx: float, cy: float) -> MotionBlob | None:
    best: MotionBlob | None = None
    best_overlap = 0.0
    for blob in blobs:
        overlap = box_overlaps_motion(cx - 1, cy - 1, cx + 1, cy + 1, [blob])
        if overlap > best_overlap:
            best_overlap = overlap
            best = blob
    return best


def motion_blob_score(
    blob: MotionBlob,
    prediction: tuple[float, float] | None,
    search_radius: float,
) -> float:
    score = math.sqrt(blob.area) * blob.compactness
    if prediction is not None:
        dist = math.hypot(blob.cx - prediction[0], blob.cy - prediction[1])
        if dist > search_radius * 1.5:
            return -1.0
        score -= dist * 2.0
    return score


def predict_on_motion_crops(
    model: YOLO,
    frame,
    blobs: list[MotionBlob],
    image_size: int,
    max_crops: int = MOTION_CROP_MAX,
) -> list[YoloDet]:
    """Run YOLO on expanded crops around motion blobs (lower conf)."""
    if not blobs:
        return []

    h, w = frame.shape[:2]
    ranked = sorted(
        blobs,
        key=lambda b: b.compactness * math.sqrt(b.area),
        reverse=True,
    )
    found: list[YoloDet] = []

    for blob in ranked[:max_crops]:
        bw = blob.x2 - blob.x1
        bh = blob.y2 - blob.y1
        pad_x = int(bw * MOTION_CROP_PAD_FRAC) + 12
        pad_y = int(bh * MOTION_CROP_PAD_FRAC) + 12
        x1 = max(0, blob.x1 - pad_x)
        y1 = max(0, blob.y1 - pad_y)
        x2 = min(w, blob.x2 + pad_x)
        y2 = min(h, blob.y2 + pad_y)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        crop = frame[y1:y2, x1:x2]
        results = model.predict(
            crop,
            conf=MOTION_CROP_CONF,
            imgsz=image_size,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            continue

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < MOTION_CROP_CONF:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            found.append(
                YoloDet(
                    track_id=None,
                    conf=conf,
                    x1=bx1 + x1,
                    y1=by1 + y1,
                    x2=bx2 + x1,
                    y2=by2 + y1,
                )
            )
    return found


def fuse_ball_detection(
    frame,
    model: YOLO,
    track_boxes,
    motion: MotionResult,
    prediction: tuple[float, float] | None,
    search_radius: float,
    active_track_id: int | None,
    kalman_initialized: bool,
    min_conf: float,
    image_size: int,
    use_motion: bool,
    enable_crop: bool = True,
    crop_max: int = MOTION_CROP_MAX,
) -> FusionPick | None:
    """
    Return a single fused football pick, or None.

    Motion gates candidates; YOLO confirms identity; Kalman guides search.
    """
    yolo_dets = parse_track_boxes(track_boxes, min_conf)
    motion_blobs = motion.blobs if use_motion and motion.ready else []

    candidates: list[tuple[float, FusionPick]] = []

    def add_candidate(score: float, pick: FusionPick) -> None:
        candidates.append((score, pick))

    def kalman_penalty(cx: float, cy: float, use_gate: bool) -> float | None:
        if not use_gate or prediction is None:
            return 0.0
        dist = math.hypot(cx - prediction[0], cy - prediction[1])
        if dist > search_radius:
            return None
        return dist * KALMAN_DIST_PENALTY

    # --- Stage 1: YOLO boxes that overlap motion (primary path) ---
    for det in yolo_dets:
        overlap = box_overlaps_motion(det.x1, det.y1, det.x2, det.y2, motion_blobs)
        if motion_blobs and overlap < MOTION_GATE_OVERLAP:
            if det.conf < HIGH_CONF_NO_MOTION:
                continue

        penalty = kalman_penalty(det.cx, det.cy, kalman_initialized)
        if penalty is None:
            continue

        blob = best_blob_for_point(motion_blobs, det.cx, det.cy)
        if blob is not None and overlap >= MOTION_MATCH_OVERLAP:
            x1, y1, x2, y2 = merge_yolo_motion_boxes(det, blob)
            source = "yolo+motion"
            overlap_boost = overlap
        else:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            source = "yolo"
            overlap_boost = overlap

        tid = det.track_id if det.track_id is not None else (active_track_id or 0)
        score = (
            det.conf * 1000.0
            + overlap_boost * MOTION_OVERLAP_WEIGHT
            - penalty
        )
        add_candidate(
            score,
            FusionPick(tid, det.cx, det.cy, x1, y1, x2, y2, det.conf, source),
        )

    # --- Stage 2: crop YOLO only when full-frame found nothing strong ---
    has_strong_yolo = any(
        p.source in ("yolo+motion", "yolo") and score > 400.0
        for score, p in candidates
    )
    if use_motion and motion_blobs and enable_crop and not has_strong_yolo:
        crop_blobs = motion_blobs
        if prediction is not None and kalman_initialized:
            near = nearest_motion_blob(motion_blobs, prediction, search_radius * 1.5)
            crop_blobs = [near] if near is not None else motion_blobs[:crop_max]
        crop_dets = predict_on_motion_crops(
            model, frame, crop_blobs, image_size, max_crops=crop_max
        )
        for det in crop_dets:
            overlap = box_overlaps_motion(det.x1, det.y1, det.x2, det.y2, motion_blobs)
            if overlap < MOTION_GATE_OVERLAP:
                continue

            penalty = kalman_penalty(det.cx, det.cy, kalman_initialized)
            if penalty is None and kalman_initialized:
                continue

            blob = best_blob_for_point(motion_blobs, det.cx, det.cy)
            if blob is not None:
                x1, y1, x2, y2 = merge_yolo_motion_boxes(det, blob)
            else:
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            tid = active_track_id or 0
            score = (
                det.conf * 900.0
                + overlap * MOTION_OVERLAP_WEIGHT
                - (penalty or 0.0)
                + 50.0
            )
            add_candidate(
                score,
                FusionPick(
                    tid, det.cx, det.cy, x1, y1, x2, y2, det.conf, "yolo+crop"
                ),
            )

    # --- Stage 3: brief motion + kalman coast (no YOLO this frame) ---
    if (
        use_motion
        and motion.ready
        and kalman_initialized
        and prediction is not None
        and active_track_id is not None
        and not any(p.source.startswith("yolo") for _, p in candidates)
    ):
        blob = nearest_motion_blob(motion_blobs, prediction, search_radius)
        if blob is not None and motion_blob_score(blob, prediction, search_radius) > 0:
            x1, y1, x2, y2 = blob_padded_box(blob)
            score = 200.0 + blob.compactness * 100.0
            add_candidate(
                score,
                FusionPick(
                    active_track_id,
                    blob.cx,
                    blob.cy,
                    x1,
                    y1,
                    x2,
                    y2,
                    MOTION_CROP_CONF * 0.9,
                    "motion+kalman",
                ),
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def fusion_pick_to_tuple(pick: FusionPick) -> tuple[int, float, float, float, float, float, float, float]:
    return (
        pick.track_id,
        pick.cx,
        pick.cy,
        pick.x1,
        pick.y1,
        pick.x2,
        pick.y2,
        pick.conf,
    )
