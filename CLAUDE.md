# Smallupload - Video Processing Pipeline

## Project Overview
GPU-accelerated pipeline (`process.py`) that converts HEVC 10-bit MKV files into HLS adaptive bitrate streams with subtitles, seek thumbnails, and gallery screenshots.

## Key Files
- `process.py` — Main pipeline script (single file, no dependencies beyond Python stdlib + FFmpeg)
- `README.md` — Full documentation
- `USAGE.txt` — Quick command reference

## Technical Stack
- Python 3.10+ (stdlib only, no pip dependencies)
- FFmpeg 8.0+ with NVENC/CUDA support
- NVIDIA GPU for hardware-accelerated encoding

## Encoding Pipeline
- **Input**: HEVC 10-bit MKVs (1080p required, 2160p optional)
- **Video**: CUDA decode → scale_cuda (format=yuv420p for 10-bit→8-bit) → h264_nvenc
- **Audio**: AAC stereo (128k for 720p/1080p, 192k for 2160p)
- **HLS**: 6s MPEG-TS segments, VOD playlist, forced IDR at GOP boundaries
- **Subtitles**: ASS → segmented WebVTT with HLS playlist
- **Thumbnails**: 160x90 JPEG every 10s with VTT metadata (CPU decode)
- **Gallery**: 10 evenly-spaced 1920x1080 WebP screenshots (~30-80 KB each)

## NVENC Settings
- Preset p6, tune hq, profile high, VBR cq 26
- Multipass fullres, rc-lookahead 20, spatial/temporal AQ (strength 8)
- B-frames 3 (middle ref), GOP 144 (6s at 23.976fps), forced-idr
- Bitrates: 720p=1500k, 1080p=3400k, 2160p=7700k

## Output Structure
```
out/
├── master.m3u8          # ABR master playlist
├── 720/index.m3u8       # 720p HLS
├── 1080/index.m3u8      # 1080p HLS
├── 2160/index.m3u8      # 2160p HLS
├── subs/index_vtt.m3u8  # Segmented WebVTT subtitles
├── thumbs/thumbs.vtt    # Seek preview thumbnails
└── gallery/             # 10x WebP gallery screenshots
```

## Important Notes
- 10-bit HEVC sources require `format=yuv420p` in scale_cuda (NVENC h264 can't encode 10-bit)
- Thumbnail generation uses CPU decode intentionally (avoids GPU→CPU transfer overhead for 1 frame/10s)
- Gallery skips first/last 5% of video to avoid intros/outros
- WebVTT timestamps from FFmpeg can be MM:SS.mmm (no hours) — parser handles both formats
- H.264 used instead of HEVC for universal browser compatibility (HEVC HLS only works in Safari)
- Branding: metadata title = "hentaiclick.tv", video stream title = "Upscaled by hentaiclick.tv"
