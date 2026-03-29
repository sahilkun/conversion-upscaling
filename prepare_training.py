#!/usr/bin/env python3
"""
Prepare GPT-SoVITS training data from labeled speaker file.
Extracts clean vocal clips per speaker with text transcripts.
"""

import os
import re
import subprocess
import sys

VOCALS_PATH = "out/dub_work/vocals.wav"
LABELS_FILE = "speaker_labels.txt"
OUTPUT_DIR = "training_data"


def parse_labels(path):
    """Parse speaker_labels.txt into list of (speaker, line_idx, timestamp_sec, text)."""
    dialogues = []

    # Parse the ASS file to get exact timestamps
    ass_lines = []
    with open("subtitle hentaiclick.ass", encoding="utf-8-sig") as f:
        for l in f:
            if not l.startswith("Dialogue:"):
                continue
            parts = l.split(",", 9)
            style = parts[3].strip()
            if style.lower() in ("sign", "credits"):
                continue
            def to_sec(ts):
                h, m, s = ts.strip().split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            start = to_sec(parts[1])
            end = to_sec(parts[2])
            text = re.sub(r"\{[^}]*\}", "", parts[9].strip())
            text = text.replace("\\N", " ").replace("\\n", " ").strip()
            if text and len(text) > 1:
                ass_lines.append({"start": start, "end": end, "text": text})

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"^([FM])(?:,\s*[FM])?\s+(\d+)\s+\[(\d+):(\d+)\]\s+(.+)", line)
            if not match:
                continue

            speaker = "female" if match.group(1) == "F" else "male"
            idx = int(match.group(2))
            text = match.group(5).strip()

            if idx < len(ass_lines):
                start = ass_lines[idx]["start"]
                end = ass_lines[idx]["end"]
                dialogues.append({
                    "speaker": speaker,
                    "idx": idx,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": text,
                })

    return dialogues


def extract_clips(dialogues):
    """Extract clean vocal clips per speaker."""
    for spk in ("female", "male"):
        spk_dir = os.path.join(OUTPUT_DIR, spk)
        os.makedirs(spk_dir, exist_ok=True)

    extracted = {"female": 0, "male": 0}
    total_dur = {"female": 0.0, "male": 0.0}

    for d in dialogues:
        spk = d["speaker"]
        clip_path = os.path.join(OUTPUT_DIR, spk, f"line_{d['idx']:03d}.wav")

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 500:
            extracted[spk] += 1
            total_dur[spk] += d["duration"]
            continue

        if d["duration"] < 1.0:
            continue

        pad = 0.1
        start = max(0, d["start"] - pad)
        length = d["duration"] + pad * 2

        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", VOCALS_PATH,
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            clip_path
        ], capture_output=True)

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 500:
            extracted[spk] += 1
            total_dur[spk] += d["duration"]

    return extracted, total_dur


def write_transcripts(dialogues):
    """Write transcript files for GPT-SoVITS training."""
    for spk in ("female", "male"):
        spk_dir = os.path.join(OUTPUT_DIR, spk)
        transcript_path = os.path.join(spk_dir, "transcript.list")

        with open(transcript_path, "w", encoding="utf-8") as f:
            for d in dialogues:
                if d["speaker"] != spk:
                    continue
                if d["duration"] < 1.0:
                    continue
                clip_path = os.path.join(spk_dir, f"line_{d['idx']:03d}.wav")
                if os.path.exists(clip_path):
                    abs_path = os.path.abspath(clip_path).replace("\\", "/")
                    f.write(f"{abs_path}|{spk}|en|{d['text']}\n")

        print(f"  {spk}: transcript at {transcript_path}")


def main():
    print("=== Preparing GPT-SoVITS Training Data ===\n")

    print("Step 1: Parsing labels...")
    dialogues = parse_labels(LABELS_FILE)
    f_count = sum(1 for d in dialogues if d["speaker"] == "female")
    m_count = sum(1 for d in dialogues if d["speaker"] == "male")
    print(f"  Female: {f_count} lines, Male: {m_count} lines\n")

    print("Step 2: Extracting vocal clips...")
    extracted, total_dur = extract_clips(dialogues)
    for spk in ("female", "male"):
        print(f"  {spk}: {extracted[spk]} clips, {total_dur[spk]:.1f}s total")
    print()

    print("Step 3: Writing transcripts...")
    write_transcripts(dialogues)
    print()

    print("=== DONE ===")
    print(f"Training data in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  female/  - {extracted['female']} clips ({total_dur['female']:.0f}s)")
    print(f"  male/    - {extracted['male']} clips ({total_dur['male']:.0f}s)")
    print()
    min_needed = 60  # GPT-SoVITS few-shot needs minimum ~1 min
    for spk in ("female", "male"):
        if total_dur[spk] >= 300:
            print(f"  {spk}: {total_dur[spk]/60:.1f} min - excellent for training")
        elif total_dur[spk] >= min_needed:
            print(f"  {spk}: {total_dur[spk]/60:.1f} min - sufficient for few-shot")
        else:
            print(f"  {spk}: {total_dur[spk]/60:.1f} min - may need more data")


if __name__ == "__main__":
    main()
