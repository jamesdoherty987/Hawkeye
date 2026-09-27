"""
stereo_record_pair.py — Record BOTH cameras at once into TWO separate MP4s.

For offline plane-calib testing. No YOLO, no overlays on the saved files —
just raw synced left/right video.

Saved as:
  exports/stereo/pairs/pair_YYYYMMDD_HHMMSS_left.mp4
  exports/stereo/pairs/pair_YYYYMMDD_HHMMSS_right.mp4

Then test with:
  python capture/stereo_plane_calib.py \\
    --video-left  exports/stereo/pairs/pair_..._left.mp4 \\
    --video-right exports/stereo/pairs/pair_..._right.mp4

Controls
────────
  R     = start / stop recording (each stop saves a new left+right pair)
  Q/ESC = quit (stops and saves if currently recording)

Usage
─────
  python capture/stereo_record_pair.py --left 0 --right 1
  python capture/stereo_record_pair.py --list
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from auto_exposure import try_set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "exports" / "stereo" / "pairs"
VIDEO_CODEC = "mp4v"
WINDOW = "Hawkeye Dual Record — R record | Q quit"
PLACEHOLDER_FPS = 30.0  # rewritten to measured fps on stop


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


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)


def make_writer(path: Path, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path}")
    return writer


def rewrite_with_fps(path: Path, fps: float) -> None:
    """Re-mux with measured FPS (OpenCV needs an FPS up front when creating)."""
    if fps <= 1.0 or not path.exists():
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return
    h, w = first.shape[:2]
    tmp = path.with_suffix(".tmp.mp4")
    try:
        writer = make_writer(tmp, w, h, fps)
    except RuntimeError:
        cap.release()
        return
    writer.write(first)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            continue
        writer.write(frame)
    writer.release()
    cap.release()
    tmp.replace(path)


def paint_preview(frame_l: np.ndarray, frame_r: np.ndarray, recording: bool, n_frames: int) -> np.ndarray:
    h = max(frame_l.shape[0], frame_r.shape[0])
    left = fit_height(frame_l, h)
    right = fit_height(frame_r, h)
    for img, title in ((left, "LEFT"), (right, "RIGHT")):
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    combo = np.hstack([left, right])
    status = "● REC" if recording else "IDLE"
    col = (0, 0, 255) if recording else (0, 200, 50)
    cv2.putText(
        combo,
        f"{status}  frames={n_frames}  |  R = record  Q = quit",
        (12, combo.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        col,
        2,
        cv2.LINE_AA,
    )
    if recording:
        cv2.rectangle(combo, (0, 0), (combo.shape[1] - 1, combo.shape[0] - 1), (0, 0, 255), 6)
    return combo


def main() -> int:
    parser = argparse.ArgumentParser(description="Record synced left+right videos as two files")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--out", type=str, default=str(OUT_DIR),
                        help=f"Output folder (default {OUT_DIR})")
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

    if args.left == args.right:
        print("ERROR: --left and --right must differ", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_l = cap_r = None
    writer_l = writer_r = None
    path_l = path_r = None
    size_l: tuple[int, int] | None = None
    size_r: tuple[int, int] | None = None
    recording = False
    n_frames = 0
    t_start = 0.0
    pairs_saved = 0

    def stop_recording() -> None:
        nonlocal writer_l, writer_r, path_l, path_r, size_l, size_r
        nonlocal recording, n_frames, pairs_saved
        if not recording:
            return
        recording = False
        elapsed = max(time.time() - t_start, 1e-3)
        fps = n_frames / elapsed
        if writer_l is not None:
            writer_l.release()
            writer_l = None
        if writer_r is not None:
            writer_r.release()
            writer_r = None
        if path_l is not None and path_r is not None and n_frames > 0:
            print(f"Rewriting with measured fps={fps:.1f} ({n_frames} frames)...")
            rewrite_with_fps(path_l, fps)
            rewrite_with_fps(path_r, fps)
            pairs_saved += 1
            print(f"Saved pair {pairs_saved}:\n  {path_l}\n  {path_r}")
            print(
                "Plane calib:\n"
                f"  python capture/stereo_plane_calib.py "
                f"--video-left {path_l} --video-right {path_r}"
            )
        elif n_frames == 0:
            print("Recording stopped with 0 frames — nothing saved.")
            if path_l is not None and path_l.exists():
                path_l.unlink(missing_ok=True)
            if path_r is not None and path_r.exists():
                path_r.unlink(missing_ok=True)
        path_l = path_r = None
        size_l = size_r = None
        n_frames = 0

    def start_recording(frame_l: np.ndarray, frame_r: np.ndarray) -> None:
        nonlocal writer_l, writer_r, path_l, path_r, size_l, size_r
        nonlocal recording, n_frames, t_start
        stop_recording()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_l = out_dir / f"pair_{ts}_left.mp4"
        path_r = out_dir / f"pair_{ts}_right.mp4"
        h_l, w_l = frame_l.shape[:2]
        h_r, w_r = frame_r.shape[:2]
        try:
            writer_l = make_writer(path_l, w_l, h_l, PLACEHOLDER_FPS)
            writer_r = make_writer(path_r, w_r, h_r, PLACEHOLDER_FPS)
        except RuntimeError as e:
            print(f"ERROR starting writers: {e}", file=sys.stderr)
            if writer_l is not None:
                writer_l.release()
                writer_l = None
            if writer_r is not None:
                writer_r.release()
                writer_r = None
            if path_l.exists():
                path_l.unlink(missing_ok=True)
            if path_r.exists():
                path_r.unlink(missing_ok=True)
            path_l = path_r = None
            return
        size_l = (w_l, h_l)
        size_r = (w_r, h_r)
        recording = True
        n_frames = 1
        t_start = time.time()
        writer_l.write(frame_l)
        writer_r.write(frame_r)
        print(f"Recording →\n  {path_l}\n  {path_r}")

    try:
        print(f"\nOpening left camera ({args.left})...")
        cap_l = open_camera(args.left)
        time.sleep(0.5)
        print(f"Opening right camera ({args.right})...")
        try:
            cap_r = open_camera(args.right)
        except RuntimeError:
            cap_l.release()
            cap_l = None
            raise
        print(f"\nPreview only until you press R. Files go to: {out_dir}\n")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 520)

        while True:
            ok_l, frame_l = read_cam(cap_l)
            ok_r, frame_r = read_cam(cap_r)
            if not ok_l or frame_l is None or not ok_r or frame_r is None:
                time.sleep(0.02)
                continue

            if (
                recording
                and writer_l is not None
                and writer_r is not None
                and size_l is not None
                and size_r is not None
                and frame_l.shape[1] == size_l[0]
                and frame_l.shape[0] == size_l[1]
                and frame_r.shape[1] == size_r[0]
                and frame_r.shape[0] == size_r[1]
            ):
                writer_l.write(frame_l)
                writer_r.write(frame_r)
                n_frames += 1

            preview = paint_preview(frame_l, frame_r, recording, n_frames)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                stop_recording()
                break
            if key in (ord("r"), ord("R")):
                if recording:
                    stop_recording()
                else:
                    start_recording(frame_l, frame_r)

    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        stop_recording()
        if cap_l is not None:
            cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {pairs_saved} pair(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
