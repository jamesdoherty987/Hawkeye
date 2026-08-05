"""
Extract still frames from recorded videos into the dataset.

Reads .mp4 files from dataset/football/videos/ and writes jpgs to
dataset/football/raw/ (same folder capture_images.py uses).

By default it saves every Nth frame so you don't get thousands of
near-identical images. Delete frames with no ball afterwards (or keep
a few empty ones as negatives).

Example:
  python capture/extract_frames.py
  python capture/extract_frames.py --every 8
  python capture/extract_frames.py --video dataset/football/videos/football_000003.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


# --- settings ---
CLASS_NAME = "football"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "dataset" / CLASS_NAME / "videos"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / CLASS_NAME / "raw"
EVERY_N_FRAMES = 8  # 8 ≈ a few frames per second from a ~30 fps clip
JPEG_QUALITY = 95


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def next_image_index(output_dir: Path, class_name: str) -> int:
    existing = list(output_dir.glob(f"{class_name}_*.jpg"))
    if not existing:
        return 1

    indices = []
    for file_path in existing:
        parts = file_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))

    return (max(indices) + 1) if indices else 1


def list_videos(video_dir: Path, specific: Path | None) -> list[Path]:
    if specific is not None:
        path = specific if specific.is_absolute() else PROJECT_ROOT / specific
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return [path]

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {video_dir}")
    return videos


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    class_name: str,
    start_index: int,
    every_n: int,
) -> tuple[int, int]:
    """Save every Nth frame. Returns (frames_saved, next_index)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Couldn't open {video_path}", file=sys.stderr)
        return 0, start_index

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(
        f"\n{video_path.name}: ~{total_frames} frames"
        + (f" @ {fps:.1f} fps" if fps > 1 else "")
        + f" — saving every {every_n} frame(s)"
    )

    frame_i = 0
    saved = 0
    index = start_index

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_i % every_n == 0:
            out_path = output_dir / f"{class_name}_{index:06d}.jpg"
            ok_write = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if ok_write:
                saved += 1
                index += 1
            else:
                print(f"Failed to write {out_path}", file=sys.stderr)

        frame_i += 1

    cap.release()
    print(f"  Saved {saved} frame(s) from {video_path.name}")
    return saved, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract training frames from videos.")
    parser.add_argument(
        "--every",
        type=int,
        default=EVERY_N_FRAMES,
        help=f"Save every Nth frame (default: {EVERY_N_FRAMES})",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional single video path. Default: all .mp4 in videos/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.every < 1:
        print("--every must be >= 1", file=sys.stderr)
        return 1

    try:
        videos = list_videos(VIDEO_DIR, args.video)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ensure_output_dir(OUTPUT_DIR)
    next_index = next_image_index(OUTPUT_DIR, CLASS_NAME)
    print(f"Writing to: {OUTPUT_DIR}")
    print(f"Starting at: {CLASS_NAME}_{next_index:06d}.jpg")
    print(f"Videos: {len(videos)}")

    total_saved = 0
    for video_path in videos:
        saved, next_index = extract_from_video(
            video_path,
            OUTPUT_DIR,
            CLASS_NAME,
            next_index,
            args.every,
        )
        total_saved += saved

    print(f"\nDone. Saved {total_saved} image(s) total.")
    print("Next: delete obvious junk / no-ball frames (keep ~10–20% empty as negatives).")
    print("Then label the keepers in Roboflow (or similar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
