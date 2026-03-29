# Fish Speech S2 Pro - RunPod Deployment

## Quick Start

### 1. Deploy on RunPod
- Go to RunPod > Deploy > GPU Pod
- Select **RTX 3090** (24GB, ~$0.46/hr)
- Click **Change Template** > enter custom Docker image:
  ```
  fishaudio/fish-speech:latest
  ```
- Set container disk to **50 GB**
- Set exposed ports: **8080, 7860**
- Enable Jupyter notebook
- Deploy

### 2. Upload files via Jupyter
Upload these to `/workspace/`:
- `audio_japanese.mp4` (your audio file)
- `subtitle.ass` (English subtitles)
- `dub_cloud.py` (the dubbing script from this folder)

### 3. Run setup (one time)
```bash
bash /workspace/setup.sh
```

### 4. Run dubbing
```bash
cd /workspace && python dub_cloud.py
```

### 5. Download result
Download `english_dub.wav` from Jupyter file browser.
Stop the pod when done.
