#!/usr/bin/env python3
"""Video processing pipeline: MKV -> HLS with subtitles and thumbnails."""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

OUT = "out"
SEGMENT_DURATION = 6
FPS = "24000/1001"
GOP = 144  # 6s at 23.976fps

# Resolution configs: (name, width, height, video_bitrate_kbps, audio_bitrate)
VARIANTS = {
    "720":  (1280,  720,  1500, "128k"),
    "1080": (1920, 1080,  3400, "128k"),
    "2160": (3840, 2160,  7700, "192k"),
}


def run(cmd, desc=None):
    """Run a command, printing it and checking for errors."""
    if desc:
        print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(f"$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def probe(path):
    """Probe a video file and return duration, framerate, stream info."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "duration": duration,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "r_frame_rate": video_stream.get("r_frame_rate", "24000/1001"),
    }


def detect_inputs():
    """Find 1080p and optional 2160p MKV files in working directory."""
    mkvs = glob.glob("*.mkv")
    src_1080 = None
    src_2160 = None
    for f in mkvs:
        low = f.lower()
        if "2160" in low or "4k" in low:
            src_2160 = f
        elif "1080" in low:
            src_1080 = f
    if not src_1080:
        print("ERROR: No 1080p MKV found (filename must contain '1080')")
        sys.exit(1)
    print(f"1080p source: {src_1080}")
    if src_2160:
        print(f"2160p source: {src_2160}")
    else:
        print("No 2160p source found; will upscale from 1080p")
    return src_1080, src_2160


def extract_subtitle(src):
    """Extract first subtitle stream as ASS."""
    if os.path.exists("subtitle.ass"):
        print("subtitle.ass already exists, skipping extraction")
        return
    run([
        "ffmpeg", "-y", "-i", src,
        "-map", "0:s:0", "-c:s", "copy",
        "subtitle.ass"
    ], "Phase 2: Extract subtitle")


def _nvenc_flags(bitrate, audio_bitrate):
    """Return common NVENC + HLS flags for a given bitrate."""
    maxrate = int(bitrate * 1.5)
    bufsize = maxrate
    return [
        "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
        "-profile:v", "high",
        "-rc:v", "vbr", "-cq", "26",
        "-b:v", f"{bitrate}k", "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        "-multipass", "fullres", "-rc-lookahead", "20",
        "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8",
        "-b_ref_mode", "middle", "-bf", "3", "-nonref_p", "1",
        "-r", FPS,
        "-g", str(GOP), "-keyint_min", str(GOP),
        "-forced-idr", "1", "-strict_gop", "1", "-no-scenecut", "1",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2",
        "-sn",
    ]


def _hls_flags(outdir):
    """Return HLS muxer flags for mpegts segments."""
    return [
        "-f", "hls",
        "-hls_time", str(SEGMENT_DURATION),
        "-hls_playlist_type", "vod",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", os.path.join(outdir, "seg_%03d.ts"),
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
    ]


def encode_hls(src, name):
    """Encode a single HLS variant using GPU-accelerated NVENC pipeline."""
    w, h, br, abr = VARIANTS[name]
    outdir = os.path.join(OUT, name)
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", src,
        "-vf", f"scale_cuda={w}:{h}:interp_algo=lanczos:format=yuv420p",
        "-aspect", "16:9",
        "-metadata", "title=hentaiclick.tv",
        "-metadata:s:v:0", "title=Upscaled by hentaiclick.tv",
    ]
    cmd += _nvenc_flags(br, abr)
    cmd += _hls_flags(outdir)
    cmd += [os.path.join(outdir, "index.m3u8")]
    run(cmd, f"Phase: Encode {name}p HLS ({w}x{h} @ {br}kbps)")


def parse_vtt_time(ts):
    """Parse a WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
    else:
        h = "0"
        m, rest = parts
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_vtt_time(seconds):
    """Format seconds as WebVTT timestamp HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def segment_subtitles(duration):
    """Convert ASS -> WebVTT, then segment into HLS-compatible VTT chunks."""
    subdir = os.path.join(OUT, "subs")
    os.makedirs(subdir, exist_ok=True)
    full_vtt = os.path.join(subdir, "full.vtt")

    # Step 1: ASS -> single WebVTT file
    run([
        "ffmpeg", "-y", "-i", "subtitle.ass",
        full_vtt
    ], "Phase 6: Segment subtitles (ASS -> VTT)")

    # Step 2: Parse cues from the VTT
    with open(full_vtt, "r", encoding="utf-8") as f:
        content = f.read()

    cues = []
    # Match VTT cue blocks: timestamp (HH:MM:SS.mmm or MM:SS.mmm) + text
    ts_pat = r"(?:\d{2}:)?\d{2}:\d{2}\.\d{3}"
    for m in re.finditer(
        rf"({ts_pat})\s*-->\s*({ts_pat})(.*?)\n(.*?)(?=\n\n|\n$|\Z)",
        content, re.DOTALL
    ):
        cues.append({
            "start": parse_vtt_time(m.group(1)),
            "end": parse_vtt_time(m.group(2)),
            "start_ts": m.group(1),
            "end_ts": m.group(2),
            "style": m.group(3),  # any positioning/style after end timestamp
            "text": m.group(4).strip()
        })

    # Step 3: Segment cues into SEGMENT_DURATION chunks
    num_segments = max(1, int(duration // SEGMENT_DURATION) + 1)
    segment_files = []
    for i in range(num_segments):
        seg_start = i * SEGMENT_DURATION
        seg_end = seg_start + SEGMENT_DURATION
        # Collect cues that overlap this segment
        seg_cues = [c for c in cues if c["end"] > seg_start and c["start"] < seg_end]
        seg_path = os.path.join(subdir, f"index{i}.vtt")
        with open(seg_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n")
            f.write(f"X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n\n")
            for c in seg_cues:
                f.write(f"{fmt_vtt_time(c['start'])} --> {fmt_vtt_time(c['end'])}\n")
                f.write(f"{c['text']}\n\n")
        segment_files.append((i, seg_path, SEGMENT_DURATION))

    # Step 4: Write m3u8 playlist
    playlist = os.path.join(subdir, "index_vtt.m3u8")
    with open(playlist, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        f.write(f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}\n")
        f.write("#EXT-X-VERSION:3\n")
        f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")
        f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
        for i in range(num_segments):
            # Last segment may be shorter
            if i == num_segments - 1:
                seg_dur = duration - i * SEGMENT_DURATION
            else:
                seg_dur = SEGMENT_DURATION
            f.write(f"#EXTINF:{seg_dur:.3f},\n")
            f.write(f"index{i}.vtt\n")
        f.write("#EXT-X-ENDLIST\n")

    os.remove(full_vtt)
    print(f"  Created {num_segments} subtitle segments")


def generate_thumbnails(src, duration):
    """Extract thumbnails and generate VTT file (CPU decode — light workload)."""
    thumbdir = os.path.join(OUT, "thumbs")
    os.makedirs(thumbdir, exist_ok=True)

    # CPU decode is optimal here: 1 frame per 10s, avoids GPU->CPU transfer overhead
    run([
        "ffmpeg", "-y",
        "-i", src,
        "-vf", "fps=1/10,scale=160:90:flags=lanczos",
        "-q:v", "5",
        os.path.join(thumbdir, "thumb%04d.jpg")
    ], "Phase 7a: Extract thumbnails")

    # Generate VTT with correct timestamps
    num_thumbs = int(duration // 10)
    if duration % 10 > 0:
        num_thumbs += 1

    lines = ["WEBVTT", ""]
    for i in range(num_thumbs):
        start = i * 10
        end = min((i + 1) * 10, duration)
        sh, sm, ss = int(start // 3600), int((start % 3600) // 60), start % 60
        eh, em, es = int(end // 3600), int((end % 3600) // 60), end % 60
        start_ts = f"{sh:02d}:{sm:02d}:{ss:06.3f}"
        end_ts = f"{eh:02d}:{em:02d}:{es:06.3f}"
        # FFmpeg thumb numbering starts at 1
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"thumb{i+1:04d}.jpg")
        lines.append("")

    vtt_path = os.path.join(thumbdir, "thumbs.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated {vtt_path} with {num_thumbs} entries")


GALLERY_COUNT = 10  # Number of gallery screenshots


def generate_gallery(src, duration):
    """Capture evenly-spaced gallery screenshots as 1080p WebP."""
    gallerydir = os.path.join(OUT, "gallery")
    os.makedirs(gallerydir, exist_ok=True)

    # Skip first/last 5% to avoid intros/outros/black frames
    margin = duration * 0.05
    usable_start = margin
    usable_end = duration - margin
    usable_duration = usable_end - usable_start
    interval = usable_duration / (GALLERY_COUNT + 1)

    for i in range(GALLERY_COUNT):
        timestamp = usable_start + interval * (i + 1)
        outpath = os.path.join(gallerydir, f"gallery_{i}.webp")
        run([
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", src,
            "-vf", "scale=1920:1080:flags=lanczos",
            "-frames:v", "1",
            "-c:v", "libwebp",
            "-quality", "80",
            "-compression_level", "6",
            outpath
        ], f"Gallery: Capture screenshot {i+1}/{GALLERY_COUNT} @ {timestamp:.1f}s")

    print(f"  Saved {GALLERY_COUNT} gallery screenshots to {gallerydir}/")


def export_upscaled(src):
    """Export a full upscaled 2160p video file (not HLS)."""
    outpath = os.path.join(OUT, "2160p_full.mkv")
    bitrate = 7700
    maxrate = int(bitrate * 1.5)
    bufsize = maxrate
    run([
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", src,
        "-vf", "scale_cuda=3840:2160:interp_algo=lanczos:format=yuv420p",
        "-aspect", "16:9",
        "-metadata", "title=hentaiclick.tv",
        "-metadata:s:v:0", "title=Upscaled by hentaiclick.tv",
        "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
        "-profile:v", "high",
        "-rc:v", "vbr", "-cq", "26",
        "-b:v", f"{bitrate}k", "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        "-multipass", "fullres", "-rc-lookahead", "20",
        "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8",
        "-b_ref_mode", "middle", "-bf", "3", "-nonref_p", "1",
        "-r", FPS,
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-sn",
        outpath
    ], "Export full upscaled 2160p video")
    print(f"  Saved {outpath}")


def write_master_playlist():
    """Write the adaptive bitrate master playlist."""
    content = """#EXTM3U
#EXT-X-VERSION:6
#EXT-X-INDEPENDENT-SEGMENTS

#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",DEFAULT=YES,AUTOSELECT=YES,LANGUAGE="en",URI="subs/index_vtt.m3u8"

#EXT-X-STREAM-INF:BANDWIDTH=1628000,RESOLUTION=1280x720,CODECS="avc1.640028,mp4a.40.2",SUBTITLES="subs"
720/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3528000,RESOLUTION=1920x1080,CODECS="avc1.640028,mp4a.40.2",SUBTITLES="subs"
1080/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=7892000,RESOLUTION=3840x2160,CODECS="avc1.640028,mp4a.40.2",SUBTITLES="subs"
2160/index.m3u8
"""
    path = os.path.join(OUT, "master.m3u8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Video processing pipeline: MKV -> HLS with subtitles and thumbnails.",
        epilog="Examples:\n"
               "  python process.py                      # run everything (HLS only, no upscale)\n"
               "  python process.py --only 720 1080      # only 720p + 1080p HLS\n"
               "  python process.py --only subs          # only subtitles\n"
               "  python process.py --only thumbs        # only thumbnails\n"
               "  python process.py --only gallery       # only gallery screenshots\n"
               "  python process.py --only upscale       # only full 2160p upscaled video\n"
               "  python process.py --only 2160 subs     # 2160p HLS + subtitles\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+",
        choices=["720", "1080", "2160", "subs", "thumbs", "gallery", "upscale"],
        help="Run only the specified tasks (default: all)"
    )
    args = parser.parse_args()

    tasks = set(args.only) if args.only else {"720", "1080", "2160", "subs", "thumbs", "gallery"}

    # Phase 0: Detect inputs
    need_video = tasks & {"720", "1080", "2160", "thumbs", "gallery", "upscale"}
    need_subs = "subs" in tasks

    if need_video:
        print("Phase 0: Detecting inputs...")
        src_1080, src_2160 = detect_inputs()

        # Phase 1: Probe video
        print("\nPhase 1: Probing video...")
        info = probe(src_1080)
        print(f"  Duration: {info['duration']:.2f}s")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  Framerate: {info['r_frame_rate']}")
    elif need_subs:
        # Subs-only still needs duration from probe
        print("Phase 0: Detecting inputs...")
        src_1080, src_2160 = detect_inputs()
        print("\nPhase 1: Probing video...")
        info = probe(src_1080)
        print(f"  Duration: {info['duration']:.2f}s")

    # Phase 2: Extract subtitle (needed for subs or any video encode that might need it)
    if need_subs:
        extract_subtitle(src_1080)

    # Phase 3-5: HLS encoding
    if "720" in tasks:
        encode_hls(src_1080, "720")
    if "1080" in tasks:
        encode_hls(src_1080, "1080")
    if "2160" in tasks:
        src_4k = src_2160 if src_2160 else src_1080
        encode_hls(src_4k, "2160")

    # Export full upscaled 2160p video
    if "upscale" in tasks:
        src_up = src_2160 if src_2160 else src_1080
        export_upscaled(src_up)

    # Phase 6: Subtitle segmentation
    if "subs" in tasks:
        segment_subtitles(info["duration"])

    # Phase 7: Thumbnails
    if "thumbs" in tasks:
        generate_thumbnails(src_1080, info["duration"])

    # Phase 7b: Gallery screenshots
    if "gallery" in tasks:
        generate_gallery(src_1080, info["duration"])

    # Phase 8: Master playlist (only when running all video encodes)
    if {"720", "1080", "2160"} <= tasks:
        print("\nPhase 8: Writing master playlist...")
        write_master_playlist()

    print(f"\nDone! Output in {OUT}/")


if __name__ == "__main__":
    main()
