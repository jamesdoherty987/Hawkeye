# Hawkeye

Final year project stuff — detecting a football (and later a sliotar) with a Raspberry Pi 4 and an Arducam UC-844 USB camera.

Using OpenCV for the camera (USB, so not Picamera2) and YOLOv8 for detection.

## Setup (Pi)

```bash
git clone https://github.com/jamesdoherty987/Hawkeye.git
cd Hawkeye
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Might also need:

```bash
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
```

## Scripts

**Test the camera / save training images**

```bash
python capture/capture_images.py
```

- SPACE = save frame → `dataset/football/raw/football_000001.jpg` etc.
- Q = quit

**Live detection (pretrained COCO model — person + sports ball)**

```bash
python capture/test_detect.py
```

First run downloads `yolov8n.pt`. Stand in front of the camera with a football. Q to quit.

If the wrong camera opens, change `CAMERA_INDEX` at the top of the script. If it's slow on the Pi, lower `IMAGE_SIZE` in `test_detect.py`.

## Folders

- `capture/` — camera + detection scripts
- `dataset/football/raw/` — images I capture
- `dataset/football/videos/` — recorded clips
- `dataset/football/yolo/` — labeled Roboflow export (train/valid)
- `dataset/sliotar/` — for later
- `training/` / `models/` / `exports/` — training output

## Train custom football model

Needs Python **3.11** venv with `requirements.txt` (not 3.14 — torch won't install).

```bash
python training/train_football.py
```

Best weights are copied to `models/football_yolov8n.pt`.

## Next steps

1. Get live detect working on the Pi
2. Collect football images / videos
3. Label in Roboflow, export YOLOv8 zip → `dataset/football/yolo/`
4. Train custom model, then detect with those weights
5. Same again for sliotar
