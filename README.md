# Smallupload

GPU-accelerated video processing pipeline that converts MKV files into HLS adaptive bitrate streams with subtitles and seek preview thumbnails.

Built for HEVC 10-bit anime content. Uses NVIDIA NVENC for hardware-accelerated encoding.

## Requirements

- **Python** 3.10+
- **FFmpeg** 7.0+ (built with `--enable-nvenc --enable-cuda-llvm --enable-cuvid`)
- **NVIDIA GPU** with NVENC support (GTX 900 series or newer)
- **NVIDIA drivers** 550+ recommended

No Python dependencies required — uses only standard library modules.

## Input Files

Place MKV files in the working directory. The script auto-detects sources by filename:

| Pattern | Role |
|---------|------|
| `*1080*`.mkv | **Required.** Used for 720p, 1080p encoding, subtitle extraction, and thumbnails |
| `*2160*`.mkv or `*4k*`.mkv | **Optional.** Used for native 2160p encoding. If absent, 2160p is upscaled from the 1080p source |

## Usage

```bash
# Run full pipeline (720p + 1080p + 2160p HLS, subtitles, thumbnails, gallery)
python process.py

# Encode specific resolutions only
python process.py --only 720 1080

# Subtitles only
python process.py --only subs

# Thumbnails only
python process.py --only thumbs

# Gallery screenshots only
python process.py --only gallery

# Export a full 2160p upscaled MKV (not HLS)
python process.py --only upscale

# Combine tasks
python process.py --only 2160 subs thumbs gallery
```

### Available Tasks

| Task | Description |
|------|-------------|
| `720` | Encode 720p HLS variant (1280x720) |
| `1080` | Encode 1080p HLS variant (1920x1080) |
| `2160` | Encode 2160p HLS variant (3840x2160) |
| `subs` | Extract and segment subtitles (ASS -> WebVTT) |
| `thumbs` | Generate seek preview thumbnails (160x90 JPEG every 10s) |
| `gallery` | Capture 10 evenly-spaced gallery screenshots (1920x1080 WebP) |
| `upscale` | Export full 2160p upscaled video as MKV |

**Default** (no `--only` flag): `720`, `1080`, `2160`, `subs`, `thumbs`, `gallery`

## Output Structure

```
out/
├── master.m3u8              # Adaptive bitrate master playlist
├── 720/
│   ├── index.m3u8           # 720p variant playlist
│   └── seg_000.ts ...       # 720p MPEG-TS segments
├── 1080/
│   ├── index.m3u8           # 1080p variant playlist
│   └── seg_000.ts ...       # 1080p MPEG-TS segments
├── 2160/
│   ├── index.m3u8           # 2160p variant playlist
│   └── seg_000.ts ...       # 2160p MPEG-TS segments
├── subs/
│   ├── index_vtt.m3u8       # Subtitle playlist
│   └── index0.vtt ...       # Segmented WebVTT files
├── thumbs/
│   ├── thumbs.vtt           # Thumbnail timing metadata
│   └── thumb0001.jpg ...    # 160x90 JPEG thumbnails
└── gallery/
    └── gallery_0.webp ...   # 1920x1080 WebP gallery screenshots (10 images)
```

## Encoding Settings

### Video (h264_nvenc)

| Setting | Value | Notes |
|---------|-------|-------|
| Preset | p6 | Best quality/speed tradeoff for NVENC |
| Tune | hq | High quality mode |
| Profile | High | H.264 High profile |
| Rate control | VBR, cq 26 | Constant quality with bitrate ceiling |
| Multipass | fullres | Two-pass encoding for better quality |
| Lookahead | 20 frames | Rate control lookahead buffer |
| Spatial AQ | On (strength 8) | Adaptive quantization for flat areas |
| Temporal AQ | On | Frame-to-frame quality smoothing |
| B-frames | 3 (middle ref) | B-frame reference mode for compression |
| GOP | 144 (6s) | Fixed GOP with forced IDR for clean HLS segments |
| Framerate | 23.976 fps | Forced output framerate |
| Pixel format | yuv420p | Converted from 10-bit source via scale_cuda |

### Bitrates

| Variant | Video | Audio | Total |
|---------|-------|-------|-------|
| 720p | 1,500 kbps (max 2,250) | 128 kbps AAC | ~1,628 kbps |
| 1080p | 3,400 kbps (max 5,100) | 128 kbps AAC | ~3,528 kbps |
| 2160p | 7,700 kbps (max 11,550) | 192 kbps AAC | ~7,892 kbps |

### HLS

| Setting | Value |
|---------|-------|
| Segment duration | 6 seconds |
| Segment format | MPEG-TS |
| Playlist type | VOD |
| HLS version | 6 |

## Pipeline Phases

The script runs through these phases in order:

1. **Detect inputs** — finds 1080p and optional 2160p MKV files
2. **Probe** — reads duration, resolution, and framerate via ffprobe
3. **Extract subtitle** — pulls first subtitle track as ASS (skipped if `subtitle.ass` exists)
4. **Encode 720p** — CUDA decode -> scale_cuda 1280x720 -> h264_nvenc -> HLS
5. **Encode 1080p** — CUDA decode -> scale_cuda 1920x1080 -> h264_nvenc -> HLS
6. **Encode 2160p** — CUDA decode -> scale_cuda 3840x2160 -> h264_nvenc -> HLS
7. **Segment subtitles** — ASS -> WebVTT -> segmented VTT chunks with HLS playlist
8. **Generate thumbnails** — extract JPEG every 10s + generate VTT timing file
9. **Generate gallery** — capture 10 evenly-spaced 1080p WebP screenshots (skips intro/outro)
10. **Write master playlist** — adaptive bitrate m3u8 linking all variants and subtitles

## Functions Reference

| Function | Description |
|----------|-------------|
| `run(cmd, desc)` | Execute a shell command with logging, raises on failure |
| `probe(path)` | Probe video file with ffprobe, returns duration/resolution/framerate |
| `detect_inputs()` | Auto-detect 1080p and 2160p MKV files in working directory |
| `extract_subtitle(src)` | Extract first subtitle stream from MKV as ASS file |
| `encode_hls(src, name)` | Encode a single HLS variant with GPU-accelerated NVENC |
| `export_upscaled(src)` | Export full 2160p upscaled MKV (non-HLS) |
| `segment_subtitles(duration)` | Convert ASS to segmented WebVTT with HLS playlist |
| `generate_thumbnails(src, duration)` | Extract seek preview JPEGs and generate VTT metadata |
| `generate_gallery(src, duration)` | Capture 10 evenly-spaced 1080p WebP gallery screenshots |
| `write_master_playlist()` | Write adaptive bitrate master m3u8 playlist |
| `parse_vtt_time(ts)` | Parse WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds |
| `fmt_vtt_time(seconds)` | Format seconds as HH:MM:SS.mmm WebVTT timestamp |

### Internal Helpers

| Function | Description |
|----------|-------------|
| `_nvenc_flags(bitrate, audio_bitrate)` | Build NVENC encoder flags for a given video/audio bitrate |
| `_hls_flags(outdir)` | Build HLS muxer flags for MPEG-TS segment output |

## Configuration

Key constants at the top of `process.py`:

```python
OUT = "out"                # Output directory
SEGMENT_DURATION = 6       # HLS segment length in seconds
FPS = "24000/1001"         # Output framerate (23.976 fps)
GOP = 144                  # GOP size (6s at 23.976 fps)
GALLERY_COUNT = 10         # Number of gallery screenshots

VARIANTS = {
    "720":  (1280,  720,  1500, "128k"),   # (width, height, video_kbps, audio_bitrate)
    "1080": (1920, 1080,  3400, "128k"),
    "2160": (3840, 2160,  7700, "192k"),
}
```

## Playback

Point any HLS-compatible player at `out/master.m3u8`:

```bash
# VLC
vlc out/master.m3u8

# ffplay
ffplay out/master.m3u8

# Web (hls.js)
# Serve the out/ directory and load master.m3u8 in an hls.js player
```

For seek preview thumbnails, use a player that supports VTT thumbnail tracks (e.g., Video.js with the thumbnails plugin) pointed at `out/thumbs/thumbs.vtt`.

## License

MIT
