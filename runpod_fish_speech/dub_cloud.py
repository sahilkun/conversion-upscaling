#!/usr/bin/env python3
"""
Fish Speech S2 Pro Cloud Dubbing Script
Runs on RunPod with Fish Speech Docker image.

Usage:
  python dub_cloud.py                          # uses defaults
  python dub_cloud.py --audio my.mp4 --subs my.ass --output dub.wav
"""

import argparse
import os
import re
import subprocess
import sys
import json
import time
import struct
import wave
import tempfile
import io
import requests

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8080"
WORKDIR = "/workspace/dub_work"
SAMPLE_RATE = 44100


# ── ASS Subtitle Parser ────────────────────────────────────────────────────
def parse_ass(path):
    """Parse ASS subtitle file, return dialogue lines."""
    dialogues = []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue

        style = parts[3].strip()
        if style.lower() in ("sign", "signs", "credits", "credit", "note"):
            continue

        def to_sec(ts):
            h, m, s = ts.strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        start = to_sec(parts[1])
        end = to_sec(parts[2])
        text = parts[9].strip()

        # Clean ASS tags
        text = re.sub(r"\{[^}]*\}", "", text)
        text = text.replace("\\N", " ").replace("\\n", " ").strip()
        if not text or len(text) < 2:
            continue

        dialogues.append({
            "start": start, "end": end,
            "duration": end - start,
            "style": style, "text": text,
        })

    dialogues.sort(key=lambda d: d["start"])
    print(f"  Parsed {len(dialogues)} dialogue lines")
    return dialogues


# ── Gender Detection ────────────────────────────────────────────────────────
def detect_gender(text):
    tl = text.lower()
    female = r"pussy|tits|my breasts|my body|cum inside me|breed me|womb|panties"
    male = r"my cock|my dick|balls|i.ll cum|ejaculate|erection"
    if re.search(female, tl):
        return "female"
    if re.search(male, tl):
        return "male"
    return None


def assign_speakers(dialogues):
    """Assign speaker IDs to dialogues."""
    for d in dialogues:
        d["gender"] = detect_gender(d["text"])

    # Fill gaps
    for i, d in enumerate(dialogues):
        if d["gender"]:
            continue
        for j in range(max(0, i - 5), min(len(dialogues), i + 5)):
            if dialogues[j]["style"] == d["style"] and dialogues[j]["gender"]:
                d["gender"] = dialogues[j]["gender"]
                break
        if not d["gender"]:
            d["gender"] = "female"

    # Assign IDs
    speaker_map = {}
    counters = {"female": 0, "male": 0}
    for d in dialogues:
        key = (d["style"], d["gender"])
        if key not in speaker_map:
            counters[d["gender"]] += 1
            speaker_map[key] = f"{d['gender']}{counters[d['gender']]}"
        d["speaker"] = speaker_map[key]

    speakers = {}
    for d in dialogues:
        speakers.setdefault(d["speaker"], 0)
        speakers[d["speaker"]] += 1
    for spk, count in speakers.items():
        print(f"  Speaker {spk}: {count} lines")
    return dialogues


# ── Fish Speech S2 Pro Emotion Tags ─────────────────────────────────────────
def add_emotion_tags(text):
    """Add Fish Speech S2 Pro inline emotion tags."""
    tl = text.lower().rstrip()

    if re.search(r"ahh|ohh|mmm|ngh|haa|cum|cumming", tl):
        return f"[moaning] {text}"
    if re.search(r"^(yes|okay|huh|please)\b", tl) and len(tl) < 20:
        return f"[whisper] {text}"
    if tl.endswith("!") and (len(tl) < 30 or re.search(r"stop|don.t|cum!|more!", tl)):
        return f"[excited] {text}"
    if re.search(r"can.t stop|so good|feels|deeper|harder|faster", tl):
        return f"[panting] {text}"
    if re.search(r"^(huh|what|eh|wait)\b", tl):
        return f"[surprised] {text}"
    if re.search(r"haha|hehe|lol|funny", tl):
        return f"[laughing] {text}"
    return text


# ── Audio Separation (Demucs) ───────────────────────────────────────────────
def separate_audio(audio_path):
    """Separate vocals from background using Demucs."""
    print("\n=== Separating vocals/background (Demucs) ===")

    wav_path = os.path.join(WORKDIR, "input.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        wav_path
    ], check=True, capture_output=True)

    subprocess.run([
        "python", "-m", "demucs", "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", WORKDIR, wav_path
    ], check=True)

    vocals = os.path.join(WORKDIR, "htdemucs", "input", "vocals.wav")
    bg = os.path.join(WORKDIR, "htdemucs", "input", "no_vocals.wav")
    print(f"  Vocals: {vocals}")
    print(f"  Background: {bg}")
    return vocals, bg


# ── Extract Reference Clips ────────────────────────────────────────────────
def extract_references(dialogues, vocals_path):
    """Extract reference clips per speaker from isolated vocals."""
    print("\n=== Extracting voice references ===")
    refs_dir = os.path.join(WORKDIR, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    MIN_DUR = 3.0

    speaker_lines = {}
    for d in dialogues:
        if d["duration"] >= MIN_DUR:
            speaker_lines.setdefault(d["speaker"], []).append(d)

    speaker_refs = {}
    for spk, lines in speaker_lines.items():
        # Pick longest calm line as reference
        best = max(lines[:8], key=lambda d: d["duration"])
        ref_path = os.path.join(refs_dir, f"{spk}.wav")

        pad = 0.3
        start = max(0, best["start"] - pad)
        length = best["duration"] + pad * 2

        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", vocals_path,
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            ref_path
        ], check=True, capture_output=True)

        speaker_refs[spk] = ref_path
        print(f"  {spk}: {best['duration']:.1f}s reference")

    return speaker_refs


# ── Generate Speech via Fish Speech API ─────────────────────────────────────
def generate_clips(dialogues, speaker_refs):
    """Generate English TTS clips using Fish Speech S2 Pro API."""
    print("\n=== Generating English speech (Fish Speech S2 Pro) ===")
    clips_dir = os.path.join(WORKDIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    fallback_ref = next(iter(speaker_refs.values())) if speaker_refs else None
    manifest = []

    for i, d in enumerate(dialogues):
        clip_path = os.path.join(clips_dir, f"line_{i:04d}.wav")

        if d["duration"] < 0.3:
            continue

        ref_wav = speaker_refs.get(d["speaker"], fallback_ref)
        if not ref_wav:
            continue

        tagged_text = add_emotion_tags(d["text"])

        try:
            # Read reference audio
            with open(ref_wav, "rb") as f:
                ref_bytes = f.read()

            # Call Fish Speech API
            files = {
                "reference_audio": ("ref.wav", ref_bytes, "audio/wav"),
            }
            data = {
                "text": tagged_text,
                "reference_text": "",  # empty = auto-detect
                "format": "wav",
                "streaming": "false",
            }

            resp = requests.post(
                f"{API_URL}/v1/tts",
                files=files,
                data=data,
                timeout=120,
            )

            if resp.status_code == 200:
                with open(clip_path, "wb") as f:
                    f.write(resp.content)
                manifest.append({
                    "path": clip_path,
                    "start": d["start"],
                    "target_duration": d["duration"],
                })
            else:
                safe = d["text"][:40]
                print(f"  WARNING: API error line {i} ({safe}): {resp.status_code}")

        except Exception as e:
            safe = d["text"][:40]
            print(f"  WARNING: Failed line {i} ({safe}): {e}")
            continue

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(dialogues)}")

    print(f"  Generated {len(manifest)} clips")
    return manifest


# ── Mix Audio ───────────────────────────────────────────────────────────────
def mix_audio(manifest, vocals_path, bg_path, total_duration, output_path):
    """Mix English TTS + ducked Japanese vocals + background."""
    print("\n=== Mixing final audio ===")

    # Create voice-only track with FFmpeg
    voice_track = os.path.join(WORKDIR, "voice_only.wav")

    # Build filter: place each clip at its timestamp
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={total_duration}"]
    filter_parts = []

    for j, m in enumerate(manifest):
        inputs.extend(["-i", m["path"]])
        delay_ms = int(m["start"] * 1000)

        # Get clip duration
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", m["path"]],
                capture_output=True, text=True
            )
            clip_dur = float(result.stdout.strip())
        except:
            clip_dur = m["target_duration"]

        target = m["target_duration"]
        ratio = clip_dur / target if clip_dur > 0 and target > 0 else 1.0

        if ratio > 1.1:
            tempo = min(ratio, 2.0)
            filter_parts.append(f"[{j+1}]atempo={tempo:.3f},adelay={delay_ms}|{delay_ms}[d{j}]")
        else:
            filter_parts.append(f"[{j+1}]adelay={delay_ms}|{delay_ms}[d{j}]")

    mix_inputs = "[0]" + "".join(f"[d{j}]" for j in range(len(manifest)))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(manifest)+1}:duration=first:dropout_transition=0[voice]")
    filter_str = ";".join(filter_parts)

    subprocess.run(
        ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[voice]",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            voice_track
        ], check=True, capture_output=True
    )

    # Build ducking filter for Japanese vocals during English speech
    duck_points = []
    for m in manifest:
        duck_points.append((m["start"], m["start"] + m["target_duration"]))

    if duck_points:
        volume_expr_parts = []
        for s, e in duck_points:
            volume_expr_parts.append(f"between(t,{s:.2f},{e:.2f})")
        duck_expr = "+".join(volume_expr_parts)
        vocal_filter = f"volume='if({duck_expr},0.15,0.8)':eval=frame"
    else:
        vocal_filter = "volume=0.8"

    # Final mix: English voice + ducked Japanese vocals + background
    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_track,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex",
        f"[0]volume=1.8[en];[1]{vocal_filter}[ja];[2]volume=0.7[bg];"
        f"[en][ja][bg]amix=inputs=3:duration=longest[out]",
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ], check=True, capture_output=True)

    print(f"  Final dub: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fish Speech S2 Pro Cloud Dubbing")
    parser.add_argument("--audio", default="/workspace/audio_japanese.mp4")
    parser.add_argument("--subs", default="/workspace/subtitle.ass")
    parser.add_argument("--output", default="/workspace/english_dub.wav")
    args = parser.parse_args()

    os.makedirs(WORKDIR, exist_ok=True)

    # Check API server
    print("Checking Fish Speech API server...")
    try:
        r = requests.get(f"{API_URL}/docs", timeout=5)
        print(f"  API server: OK")
    except:
        print("  ERROR: API server not running. Run setup.sh first!")
        sys.exit(1)

    # Get audio duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", args.audio],
        capture_output=True, text=True
    )
    total_duration = float(result.stdout.strip())
    print(f"Audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Step 1: Parse subtitles
    print("\n=== Step 1: Parse subtitles ===")
    dialogues = parse_ass(args.subs)

    # Step 2: Assign speakers
    print("\n=== Step 2: Assign speakers ===")
    dialogues = assign_speakers(dialogues)

    # Step 3: Separate audio
    vocals, bg = separate_audio(args.audio)

    # Step 4: Extract references
    speaker_refs = extract_references(dialogues, vocals)

    # Step 5: Generate TTS
    manifest = generate_clips(dialogues, speaker_refs)

    # Step 6: Mix
    mix_audio(manifest, vocals, bg, total_duration, args.output)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n=== DONE! Output: {args.output} ({size_mb:.1f} MB) ===")
    print("Download it from Jupyter file browser, then STOP the pod!")


if __name__ == "__main__":
    main()
