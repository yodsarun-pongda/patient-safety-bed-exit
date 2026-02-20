# Bed Sit Alert (Sit vs Sleep) — 3-Week MVP

This repo is a **two-part** project:
1) **Model**: image classification from video frames (`sit` vs `sleep`)
2) **Application**: backend API + LINE OA notification (or a stub) for real-time alerts

> Goal: When an elderly/patient **sits up on the bed**, the system **alerts a phone immediately**.

---

## Project Structure

```
bed-sit-alert/
  model/
    data/
      raw_videos/        # put your videos here (not committed)
      frames/            # extracted frames, labeled into sit/ sleep
      splits/            # train/val/test split folders
    scripts/
      extract_frames.py  # video -> frames
      split_dataset.py   # frames -> splits
    train/
      train.py           # transfer learning (MobileNetV3 / ResNet18)
      eval.py            # metrics + confusion matrix
      export.py          # export TorchScript
    infer/
      infer_video.py     # run inference on a video (with debounce)
      emit_event.py      # send event to backend
    configs/
      config.yaml        # thresholds, fps, debounce, cooldown
    requirements.txt
  backend/
    app/
      main.py            # FastAPI server
      routes.py          # /event endpoint
      notify_line.py     # LINE OA push helper (optional)
      storage.py         # sqlite event log
    requirements.txt
    .env.example
  docs/
    DATASET_GUIDE.md
    EVALUATION.md
    ARCHITECTURE.md
  Makefile
```

---

## Quickstart (Recommended)

### 0) Create environments
**Model**
```bash
cd model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Backend**
```bash
cd ../backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## 1) Data: Video → Frames

Put videos in:
```
model/data/raw_videos/
```
Extract frames (default 1 fps):
```bash
python model/scripts/extract_frames.py --input model/data/raw_videos --output model/data/frames_raw --fps 1
```

### Labeling (fastest approach for 3 weeks)
Create two folders and **move/copy** frames into them:
```
model/data/frames/sit/
model/data/frames/sleep/
```
You can do it manually by file browser (fast) or by helper notes in:
- `docs/DATASET_GUIDE.md`

---

## 2) Split dataset

```bash
python model/scripts/split_dataset.py  --input model/data/frames  --output model/data/splits --train 0.8 --val 0.1 --test 0.1
```

---

## 3) Train (Transfer Learning)
3.1 Allow permission access data
```
chmod +x "venv/lib/python3.10/site-packages/torch/bin/torch_shm_manager"
```
3.2 Then follow command as below
```bash
python model/train/train_backup.py   --data model/data/splits  --arch mobilenet_v3_small   --epochs 10   --batch 32
```

Evaluate:
```bash
python train/eval.py --data data/splits --ckpt model_ckpt.pt
```

Export TorchScript:
```bash
python train/export.py --ckpt model_ckpt.pt --out sit_sleep.ts
```

---

## 4) Run Backend (Alert API)

Edit `.env` (LINE is optional; you can start with a stub log-only mode):
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test event:
```bash
curl -X POST http://localhost:8000/event   -H "Content-Type: application/json"   -d '{"event":"SIT_DETECTED","confidence":0.92,"source":"demo","timestamp":"2026-02-08T21:00:00"}'
```

---

## 5) Run Inference on a Video (with debounce + cooldown)

```bash
cd model
python infer/infer_video.py   --video data/raw_videos/demo.mp4   --model sit_sleep.ts   --backend http://localhost:8000/event
```

The inference runner uses:
- **Debounce**: require `SIT` prediction for N consecutive frames
- **Cooldown**: after alert, wait M seconds before sending another alert

Tune these in:
- `model/configs/config.yaml`

---

## What to implement next (for Codex IDE)
1) Improve labeling workflow (semi-automatic UI)
2) Add augmentation and class imbalance handling
3) Add confidence calibration + optimal threshold search
4) Add real-time camera (USB/RTSP) support
5) Add LINE OA push credentials and real push notification

---

## Notes
- This repo is an MVP scaffold. You should add your real videos and labels locally.
- For privacy, avoid storing full videos; keep only event logs and optional snapshots.


## Manual test
python model/infer/predict_sit_sleep.py --ckpt artifacts/model_ckpt.pt --image test_image/image.png


## Manual Test
python model/infer/predict_skeleton_sit_sleep.py --ckpt artifacts/skeleton_model_ckpt.pt --image test_image/image.png  --save artifacts/predict_skeleton_demo_2.jpg