"""
Play a recorded video at half speed and save frames with SPACE.

Useful when you want to pick only the good ball frames by hand.

SPACE = save current frame → dataset/football/raw/
A     = jump back 1 seconds
D     = jump forward 1 seconds
P     = pause / resume
Q     = quit

Example:
  python capture/review_video.py dataset/football/videos/football_000003.mp4
  python capture/review_video.py dataset/football/videos/football_000003.mp4 --speed 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


# --- settings ---
CLASS_NAME = "football"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / CLASS_NAME / "raw"
DEFAULT_SPEED = 0.5
SEEK_SECONDS = 1.0
JPEG_QUALITY = 95
WINDOW_NAME = "Hawkeye Review — SPACE save | A/D seek | P pause | Q quit"


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


def resolve_video_path(raw: Path) -> Path:
    path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if not path.exists():
        alt = PROJECT_ROOT / raw
        if alt.exists():
            return alt.resolve()
        raise FileNotFoundError(f"Video not found: {raw}")
    return path


def draw_overlay(
    frame,
    frame_index: int,
    total_frames: int,
    saved_count: int,
    paused: bool,
    speed: float,
):
    display = frame.copy()
    status = "PAUSED" if paused else f"PLAY {speed:.2f}x"
    colour = (0, 255, 255) if paused else (0, 255, 0)

    lines = [
        f"{status}  frame {frame_index}/{total_frames}",
        f"Saved this session: {saved_count}",
        "SPACE = save | A/D = +/-1s | P = pause | Q = quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
        y += 28

    return display


def save_frame(frame, output_dir: Path, class_name: str, index: int) -> Path | None:
    output_path = output_dir / f"{class_name}_{index:06d}.jpg"
    ok = cv2.imwrite(
        str(output_path),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        print(f"Failed to save {output_path}", file=sys.stderr)
        return None
    return output_path


def review_video(video_path: Path, output_dir: Path, class_name: str, speed: float) -> int:
    ensure_output_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    delay_ms = max(1, int(1000.0 / (fps * speed)))
    seek_frames = max(1, int(round(fps * SEEK_SECONDS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    next_index = next_image_index(output_dir, class_name)
    saved_count = 0
    paused = False
    frame_index = 0
    frame = None

    print(f"Video: {video_path}")
    print(f"Source FPS: {fps:.1f} | playback: {speed}x | frame delay: {delay_ms} ms")
    print(f"Arrow seek: {SEEK_SECONDS:.0f}s ({seek_frames} frames)")
    print(f"Saving to: {output_dir}")
    print(f"Starting at: {class_name}_{next_index:06d}.jpg")
    print("SPACE = save | A/D = +/-1s | P = pause | Q = quit")

    def seek(delta_frames: int) -> None:
        nonlocal frame, frame_index

        # frame_index is 1-based for the currently displayed frame.
        current_zero_based = max(frame_index - 1, 0)
        target = current_zero_based + delta_frames
        target = max(0, target)
        if total_frames > 0:
            target = min(target, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, next_frame = cap.read()
        if not ok or next_frame is None:
            print(f"Couldn't seek to frame {target + 1}", file=sys.stderr)
            return

        frame = next_frame
        frame_index = target + 1
        direction = "forward" if delta_frames > 0 else "back"
        print(f"Jumped {direction} to frame {frame_index}/{total_frames}")

    while True:
        if not paused or frame is None:
            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                print("End of video.")
                break
            frame = next_frame
            frame_index += 1

        display = draw_overlay(
            frame, frame_index, total_frames, saved_count, paused, speed
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

        if key in (ord("a"), ord("A")):
            seek(-seek_frames)
        elif key in (ord("d"), ord("D")):
            seek(seek_frames)
        elif key == ord(" "):
            path = save_frame(frame, output_dir, class_name, next_index)
            if path is not None:
                print(f"Saved {path.name} (video frame {frame_index})")
                next_index += 1
                saved_count += 1
        elif key in (ord("p"), ord("P")):
            paused = not paused
            print("Paused." if paused else "Playing.")
        elif key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a video slowly and save frames with SPACE."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the .mp4 to review",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Playback speed (default: {DEFAULT_SPEED})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.speed <= 0:
        print("--speed must be > 0", file=sys.stderr)
        return 1

    try:
        video_path = resolve_video_path(args.video)
        saved = review_video(video_path, OUTPUT_DIR, CLASS_NAME, args.speed)
        print(f"Done. Saved {saved} frame(s).")
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
