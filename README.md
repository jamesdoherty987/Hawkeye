# Hawkeye

Final year project — detecting a football (and later a sliotar) with a camera + YOLOv8.

Works with an Arducam UC-844 USB camera on Mac, Windows, or Raspberry Pi (OpenCV, not Picamera2).

## Setup

Use **Python 3.11** (3.14 cannot install the pinned torch build).

```bash
git clone https://github.com/jamesdoherty987/Hawkeye.git
cd Hawkeye
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Pi you might also need:

```bash
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
```

Copy `models/football_yolov8n.pt` onto the machine if you already trained (do not retrain just to move).

## Scripts

**Record video (pitch / outdoor)**

```bash
python capture/record_video.py
```

- R = start/stop recording → `dataset/football/videos/`
- Q = quit
- Exposure auto-adjusts from scene brightness

**Review a video and save ball frames by hand**

```bash
python capture/review_video.py dataset/football/videos/football_000003.mp4
```

- SPACE = save frame → `dataset/football/raw/`
- A / D = back / forward 3 seconds
- P = pause | Q = quit

**Auto-extract every Nth frame from all videos**

```bash
python capture/extract_frames.py
python capture/extract_frames.py --every 8
```

**Still capture (SPACE to save)**

```bash
python capture/capture_images.py
```

**Live detect with your trained model**

```bash
python capture/test_detect.py
```

- Loads `models/football_yolov8n.pt`
- Draws a box on the ball
- Saves every frame with a ball to `dataset/football/detections/`
- Q = quit

**Detect + track path (Phase 1 — single camera)**

```bash
python capture/track_video.py dataset/football/videos/football_000003.mp4
python capture/track_video.py dataset/football/videos/football_000003.mp4 --save-video
python capture/track_video.py --camera
```

Uses **YOLO + ByteTrack** (built into Ultralytics). Draws a trail, speed (px/s), and direction. Saves path CSV to `exports/football/paths/`. Annotated video (optional) → `exports/football/tracked/`.

If the wrong camera opens, use `--camera-index 1` (track) or change `CAMERA_INDEX` in `test_detect.py`.

**Train (after Roboflow YOLOv8 export is in `dataset/football/yolo/`)**

```bash
python training/train_football.py
```

Best weights → `models/football_yolov8n.pt`

## Folders

- `capture/` — camera + detection scripts
- `dataset/football/raw/` — hand-picked / extracted stills
- `dataset/football/videos/` — recorded clips
- `dataset/football/yolo/` — labeled Roboflow export (train/valid)
- `dataset/football/detections/` — frames saved when live detect sees a ball
- `exports/football/tracked/` — annotated videos with ball trail
- `exports/football/paths/` — CSV path data (frame, time, x, y)
- `dataset/sliotar/` — for later
- `training/` / `models/` / `exports/` — training output

## Data collection (what goes where)

| Goal | What to collect | Where it goes | Then |
|------|-----------------|---------------|------|
| **Train detector** | Ball in frame (still + moving) | `dataset/football/raw/` via review/capture | Label in Roboflow → export to `dataset/football/yolo/` → retrain |
| **Hard negatives** | Heads, cups, empty pitch — **no box** | Same Roboflow project, upload, leave unlabeled | Retrain — reduces false positives |
| **Test tracking** | Clips with ball crossing the frame | `dataset/football/videos/` | Run `track_video.py` on them |
| **More moving ball** | Kicks, rolls, far + close | Record → review video → save key frames **and** keep full videos for tracking | Both labeling and `track_video.py` |

You do **not** need 1000 images. Aim for ~200–400 labeled ball images + ~50–100 negatives, plus a few good pitch videos for tracking tests.

Tracking quality depends on detection — improve the model first if the trail keeps breaking.

## Next steps

1. Improve dataset (negatives + varied ball shots) and retrain
2. Run `track_video.py` on your pitch videos; tune `CONFIDENCE` in `track_video.py`
3. Same pipeline for sliotar later
4. Phase 2: camera calibration + ground plane; Phase 3: second camera + 3D
