#!/usr/bin/env python3
"""
Assemble Fish Speech TTS clips into a dubbed audio track.

1. Parse subtitle.ass for dialogue timings
2. Place each line_NNNN.wav at its subtitle start time on a silent timeline
3. Mix English voice + ducked Japanese vocals + background music via FFmpeg

Called by process.py:
  py -3.11 dub_generate.py subtitle.ass 990.7 out source.mkv [voices/]
"""

import argparse
import os
import sys

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dub_common


def main():
    parser = argparse.ArgumentParser(description="Assemble Fish Speech TTS clips into dubbed audio")
    parser.add_argument("ass_path", help="Path to subtitle.ass")
    parser.add_argument("duration", type=float, help="Total duration in seconds")
    parser.add_argument("out_dir", help="Output directory (e.g., 'out')")
    parser.add_argument("src_mkv", help="Source MKV file")
    parser.add_argument("voices_dir", nargs="?", default="voices",
                        help="Directory with default voice WAVs")
    args = parser.parse_args()

    workdir = os.path.join(args.out_dir, "dub_work")
    clips_dir = os.path.join(workdir, "fish_clips")
    vocals = os.path.join(workdir, "vocals.wav")
    bg = os.path.join(workdir, "background.wav")
    output = os.path.join(workdir, "english_dub.wav")

    # Use actual duration from arg (never hardcoded)
    total_duration = args.duration
    if total_duration <= 0:
        total_duration = dub_common.probe_duration(args.src_mkv)

    # Parse subtitles
    print("Parsing subtitles...")
    subs = dub_common.parse_ass(args.ass_path)

    # Count available TTS clips
    if not os.path.isdir(clips_dir):
        print(f"ERROR: No TTS clips directory found at {clips_dir}")
        print("       Run Fish Speech TTS generation first (cloud_dub.py)")
        sys.exit(1)

    num_clips = len([f for f in os.listdir(clips_dir)
                     if f.startswith("line_") and f.endswith(".wav")])
    print(f"  Found {num_clips} TTS clips")

    lines_to_use = min(len(subs), num_clips)

    # Build manifest for assembly
    manifest = []
    for i in range(lines_to_use):
        clip_path = os.path.join(clips_dir, f"line_{i:04d}.wav")
        if not os.path.exists(clip_path):
            continue

        emotion, speed_factor, volume_factor = dub_common.classify_emotion(subs[i]["text"])
        manifest.append({
            "path": clip_path,
            "start": subs[i]["start"],
            "target_duration": subs[i]["duration"],
            "emotion": emotion,
            "volume_factor": volume_factor,
        })

    print(f"  Will assemble {len(manifest)} clips")

    # Assemble voice track
    voice_path = dub_common.assemble_voice_track(manifest, total_duration, workdir)

    # Ensure Demucs separated audio exists
    if not os.path.exists(vocals) or not os.path.exists(bg):
        print("  Running Demucs separation...")
        vocals, bg = dub_common.separate_audio(args.src_mkv, workdir)

    # Mix final audio
    dub_common.mix_audio(voice_path, vocals, bg, manifest, output)

    print(f"\nDone! Output: {output}")


if __name__ == "__main__":
    main()
