#!/usr/bin/env python3
"""
Fish Speech 1.5 Local Dubbing Script — v2 (with XTTS pipeline improvements)
Runs locally with Fish Speech API server on RTX 3050.

Features ported from XTTS pipeline:
  - Calmness-scored reference selection
  - Per-speaker consistent voice (same ref for all lines of a speaker)
  - Emotion-aware post-processing (pitch/volume/speed per emotion)
  - Smart vocal ducking (keeps moaning/breathing between dialogue)
  - Smooth duck transitions (fade in/out)

Usage:
  py -3.11 cloud_dub.py
  py -3.11 cloud_dub.py --src input.mkv --subs subtitle.ass --output english_dub.wav
"""

import argparse
import array
import base64
import os
import re
import struct
import subprocess
import sys
import wave
import requests

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dub_common

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8080"
WORKDIR = os.path.join("out", "dub_work")
SAMPLE_RATE = 44100

# Fish Speech consistency settings — lower temp + fixed seed + memory cache
# reduces voice drift between API calls
TTS_SEED = 42
TTS_TEMPERATURE = 0.4
TTS_TOP_P = 0.7
TTS_REPETITION_PENALTY = 1.2


# ── ASS Subtitle Parser ────────────────────────────────────────────────────
def parse_ass(path):
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


# ── Gender Detection + Speaker Assignment ───────────────────────────────────
def detect_gender(text):
    tl = text.lower()
    if re.search(r"pussy|tits|my breasts|my body|cum inside me|breed me|womb|panties", tl):
        return "female"
    if re.search(r"my cock|my dick|balls|i.ll cum|ejaculate|erection", tl):
        return "male"
    return None


def assign_speakers(dialogues):
    for d in dialogues:
        d["gender"] = detect_gender(d["text"])

    for i, d in enumerate(dialogues):
        if d["gender"]:
            continue
        for j in range(max(0, i - 5), min(len(dialogues), i + 5)):
            if dialogues[j]["style"] == d["style"] and dialogues[j]["gender"]:
                d["gender"] = dialogues[j]["gender"]
                break
        if not d["gender"]:
            d["gender"] = "female"

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


# ── Emotion Classification (ported from XTTS _classify_emotion) ────────────
def classify_emotion(text):
    """Classify dialogue line emotion. Returns (emotion, speed_factor, volume_factor).

    emotion: tag for Fish Speech + post-processing type
    speed_factor: playback speed adjustment (>1 = faster)
    volume_factor: volume multiplier
    """
    tl = text.lower().rstrip()

    # Moaning / climax
    if re.search(r"ahh|ohh|mmm|ngh|haa|cumming|cum!|shooting out", tl):
        return "moaning", 0.95, 1.3

    # Exclamatory / intense short bursts
    if tl.endswith("!") and len(tl) < 25:
        return "exclaim", 1.1, 1.3

    # Whisper / soft
    if re.search(r"^(yes|okay|huh|please)\b", tl) and len(tl) < 20:
        return "whisper", 0.9, 0.7

    # Questions
    if tl.endswith("?"):
        return "question", 1.0, 1.0

    # Intense / passionate
    if re.search(r"can.t stop|can.t wait|so bad|please.*cum|breed me|don.t pull|fill my|deeper|harder|faster|so good|feels", tl):
        return "intense", 1.05, 1.2

    # Surprised
    if re.search(r"^(huh|what|eh|wait)\b", tl):
        return "surprised", 1.0, 1.1

    # Laughing
    if re.search(r"haha|hehe|lol|funny", tl):
        return "laughing", 1.0, 1.0

    # Normal
    return "neutral", 1.0, 1.0


def add_emotion_tags(text, emotion):
    """Add Fish Speech inline emotion tags based on classified emotion."""
    tag_map = {
        "moaning": "[moaning]",
        "exclaim": "[excited]",
        "whisper": "[whisper]",
        "intense": "[excited]",
        "surprised": "[surprised]",
        "laughing": "[laughing]",
        "question": "",
        "neutral": "",
    }
    tag = tag_map.get(emotion, "")
    if tag:
        return f"{tag} {text}"
    return text


# ── Calmness Measurement (ported from XTTS _measure_calmness) ──────────────
def measure_calmness(wav_path):
    """Measure how calm a vocal clip is. Lower score = calmer.

    Uses FFmpeg astats to get RMS level variance. Calm/neutral speech has
    consistent energy, expressive speech has wild swings.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", wav_path, "-af",
             "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
             "-f", "null", "-"],
            capture_output=True, text=True
        )
        rms_values = []
        for line in result.stderr.split("\n"):
            if "RMS_level" in line and "=" in line:
                try:
                    val = float(line.split("=")[-1].strip())
                    if val > -100:
                        rms_values.append(val)
                except ValueError:
                    pass
        if len(rms_values) < 2:
            return None
        mean = sum(rms_values) / len(rms_values)
        variance = sum((v - mean) ** 2 for v in rms_values) / len(rms_values)
        return variance
    except Exception:
        return None


# ── Audio Separation (Demucs) ───────────────────────────────────────────────
def separate_audio(src_mkv):
    print("\n=== Step 3: Separating vocals/background (Demucs) ===")

    vocals = os.path.join(WORKDIR, "vocals.wav")
    bg = os.path.join(WORKDIR, "background.wav")

    if os.path.exists(vocals) and os.path.exists(bg):
        print("  Cached, skipping.")
        return vocals, bg

    full_audio = os.path.join(WORKDIR, "full_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", src_mkv,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        full_audio
    ], check=True, capture_output=True)

    separated_dir = os.path.join(WORKDIR, "separated")
    subprocess.run([
        sys.executable, "-m", "demucs", "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", separated_dir, full_audio
    ], check=True)

    src_vocals = os.path.join(separated_dir, "htdemucs", "full_audio", "vocals.wav")
    src_bg = os.path.join(separated_dir, "htdemucs", "full_audio", "no_vocals.wav")

    import shutil
    shutil.move(src_vocals, vocals)
    shutil.move(src_bg, bg)
    print(f"  Vocals: {vocals}")
    print(f"  Background: {bg}")
    return vocals, bg


# ── Extract CALM References per Speaker (ported from XTTS) ─────────────────
def extract_references(dialogues, vocals_path):
    """Extract and CONCATENATE the calmest vocal clips per speaker into a long reference.

    Short references (~3s) cause Fish Speech to re-interpret the voice each call,
    leading to inconsistent output. By concatenating the top N calmest clips into
    a single 10-15s reference, the model gets a much more stable voice embedding.

    Steps:
    1. Extract up to 12 candidate clips per speaker (>= 2s duration)
    2. Score each by calmness (RMS variance — lower = calmer)
    3. Concatenate the top calmest clips until we hit ~12s total
    4. Use that long composite clip as THE reference for all lines
    """
    print("\n=== Step 4: Extracting calm voice references (long composite) ===")
    refs_dir = os.path.join(WORKDIR, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    MIN_DUR = 2.0
    MAX_CANDIDATES = 12
    TARGET_REF_SECONDS = 12.0

    # Group candidates per speaker
    speaker_candidates = {}
    for i, d in enumerate(dialogues):
        if d["duration"] < MIN_DUR:
            continue
        spk = d["speaker"]
        if spk not in speaker_candidates:
            speaker_candidates[spk] = []
        if len(speaker_candidates[spk]) < MAX_CANDIDATES:
            speaker_candidates[spk].append((i, d))

    speaker_refs = {}
    for spk, candidates in speaker_candidates.items():
        print(f"  {spk}: evaluating {len(candidates)} candidates...")

        # Extract and score all candidates
        scored = []
        for idx, (i, d) in enumerate(candidates):
            ref_path = os.path.join(refs_dir, f"{spk}_cand{idx}.wav")
            pad = 0.3
            start = max(0, d["start"] - pad)
            length = d["duration"] + pad * 2

            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                "-i", vocals_path,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                ref_path
            ], check=True, capture_output=True)

            score = measure_calmness(ref_path)
            if score is not None:
                scored.append((score, ref_path, d["duration"]))

        if not scored:
            # Fallback: use first candidate raw
            first_path = os.path.join(refs_dir, f"{spk}_cand0.wav")
            final_ref = os.path.join(refs_dir, f"{spk}.wav")
            if os.path.exists(first_path):
                import shutil
                shutil.copy2(first_path, final_ref)
                speaker_refs[spk] = final_ref
                print(f"    -> Fallback (first candidate, no calmness scores)")
            continue

        # Sort by calmness (lower = calmer) and pick clips until ~TARGET_REF_SECONDS
        scored.sort(key=lambda x: x[0])
        chosen = []
        total_dur = 0.0
        for score, path, dur in scored:
            chosen.append(path)
            total_dur += dur
            if total_dur >= TARGET_REF_SECONDS:
                break

        final_ref = os.path.join(refs_dir, f"{spk}.wav")

        if len(chosen) == 1:
            import shutil
            shutil.copy2(chosen[0], final_ref)
            print(f"    -> Single clip ({total_dur:.1f}s, calmness: {scored[0][0]:.2f})")
        else:
            # Concatenate chosen clips with 0.3s silence between them
            concat_list = os.path.join(refs_dir, f"{spk}_concat.txt")
            silence_path = os.path.join(refs_dir, "silence_0.3s.wav")

            # Generate a short silence separator
            if not os.path.exists(silence_path):
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=mono", "-t", "0.3",
                    "-acodec", "pcm_s16le", silence_path
                ], check=True, capture_output=True)

            with open(concat_list, "w", encoding="utf-8") as f:
                for ci, clip in enumerate(chosen):
                    p = os.path.abspath(clip).replace("\\", "/").replace("'", "\\'")
                    f.write(f"file '{p}'\n")
                    if ci < len(chosen) - 1:
                        sp = os.path.abspath(silence_path).replace("\\", "/").replace("'", "\\'")
                        f.write(f"file '{sp}'\n")

            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                final_ref
            ], check=True, capture_output=True)

            os.remove(concat_list)
            print(f"    -> Composite: {len(chosen)} clips, ~{total_dur:.1f}s (calmness range: {scored[0][0]:.2f}-{scored[len(chosen)-1][0]:.2f})")

        speaker_refs[spk] = final_ref

    # Clean up candidate files
    for f in os.listdir(refs_dir):
        if "_cand" in f:
            os.remove(os.path.join(refs_dir, f))
    silence = os.path.join(refs_dir, "silence_0.3s.wav")
    if os.path.exists(silence):
        os.remove(silence)

    print(f"  References: {', '.join(f'{s} -> {os.path.basename(p)}' for s, p in speaker_refs.items())}")
    return speaker_refs


# ── Two-pass bootstrap: generate English refs from Japanese vocals ──────────
def bootstrap_english_refs(dialogues, speaker_refs):
    """Two-pass voice bootstrapping for consistent English output.

    Problem: Using Japanese vocal clips as reference for English TTS causes
    voice inconsistency — the model re-generates a different voice embedding
    each call because it's cloning across languages.

    Fix:
    1. Pass 1: For each speaker, generate ONE English clip using Japanese ref
    2. Pass 2: Use that English clip as the reference for ALL remaining lines
    This bootstraps a stable English voice from each character.
    """
    print("\n=== Step 4b: Bootstrapping English voice references ===")
    refs_dir = os.path.join(WORKDIR, "refs")
    cache = {}

    for spk, jp_ref_path in speaker_refs.items():
        en_ref_path = os.path.join(refs_dir, f"{spk}_english.wav")

        # Skip if already bootstrapped
        if os.path.exists(en_ref_path) and os.path.getsize(en_ref_path) > 1000:
            with open(en_ref_path, "rb") as f:
                cache[spk] = base64.b64encode(f.read()).decode()
            print(f"  {spk}: cached English ref (existing)")
            continue

        # Find a good neutral line for this speaker to bootstrap
        boot_text = None
        for d in dialogues:
            if d["speaker"] == spk and d["duration"] >= 2.0:
                emo, _, _ = classify_emotion(d["text"])
                if emo == "neutral":
                    boot_text = d["text"]
                    break
        if not boot_text:
            # Fallback: use first line >= 2s
            for d in dialogues:
                if d["speaker"] == spk and d["duration"] >= 2.0:
                    boot_text = d["text"]
                    break
        if not boot_text:
            boot_text = "This is a test of the voice cloning system."

        # Pass 1: Generate multiple English clips from Japanese reference, concatenate
        with open(jp_ref_path, "rb") as f:
            jp_b64 = base64.b64encode(f.read()).decode()

        # Find 2-3 neutral lines for this speaker to bootstrap a longer ref
        boot_lines = []
        for d in dialogues:
            if d["speaker"] == spk and d["duration"] >= 2.0:
                emo, _, _ = classify_emotion(d["text"])
                if emo == "neutral":
                    boot_lines.append(d["text"])
                    if len(boot_lines) >= 3:
                        break
        if not boot_lines:
            boot_lines = [boot_text]

        print(f"  {spk}: bootstrapping with {len(boot_lines)} lines...")

        try:
            en_clips = []
            for bl in boot_lines:
                best_content = None
                best_size = 0
                for attempt in range(2):
                    resp = requests.post(
                        f"{API_URL}/v1/tts",
                        json={
                            "text": bl, "reference_audio": jp_b64, "format": "wav",
                            "seed": TTS_SEED, "temperature": TTS_TEMPERATURE,
                            "top_p": TTS_TOP_P, "repetition_penalty": TTS_REPETITION_PENALTY,
                            "use_memory_cache": "on",
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200 and len(resp.content) > best_size:
                        best_content = resp.content
                        best_size = len(resp.content)
                if best_content and best_size > 1000:
                    en_clips.append(best_content)

            if en_clips:
                if len(en_clips) == 1:
                    # Single clip
                    with open(en_ref_path, "wb") as f:
                        f.write(en_clips[0])
                else:
                    # Concatenate multiple English clips into a longer reference
                    refs_dir = os.path.join(WORKDIR, "refs")
                    tmp_paths = []
                    for ci, clip_data in enumerate(en_clips):
                        tmp = os.path.join(refs_dir, f"{spk}_enboot{ci}.wav")
                        with open(tmp, "wb") as f:
                            f.write(clip_data)
                        tmp_paths.append(tmp)

                    concat_list = os.path.join(refs_dir, f"{spk}_enconcat.txt")
                    with open(concat_list, "w", encoding="utf-8") as f:
                        for tp in tmp_paths:
                            p = os.path.abspath(tp).replace("\\", "/").replace("'", "\\'")
                            f.write(f"file '{p}'\n")

                    subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list,
                        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                        en_ref_path
                    ], check=True, capture_output=True)

                    # Cleanup temp files
                    os.remove(concat_list)
                    for tp in tmp_paths:
                        os.remove(tp)

                with open(en_ref_path, "rb") as f:
                    cache[spk] = base64.b64encode(f.read()).decode()
                size_kb = os.path.getsize(en_ref_path) / 1024
                print(f"    -> English ref created ({len(en_clips)} clips, {size_kb:.0f} KB)")
            else:
                # Fallback to Japanese ref
                cache[spk] = jp_b64
                print(f"    -> Fallback to Japanese ref")

        except Exception as e:
            cache[spk] = jp_b64
            print(f"    -> Error ({e}), using Japanese ref")

    return cache


# ── Generate Speech via Fish Speech API ─────────────────────────────────────
def _wav_duration(path):
    """Return duration of a WAV file in seconds."""
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _tts_request(tagged_text, ref_b64, temperature, repetition_penalty, speed=1.0):
    """Single TTS API call. Returns response or raises."""
    return requests.post(
        f"{API_URL}/v1/tts",
        json={
            "text": tagged_text, "reference_audio": ref_b64, "format": "wav",
            "seed": TTS_SEED, "temperature": temperature,
            "top_p": TTS_TOP_P, "repetition_penalty": repetition_penalty,
            "use_memory_cache": "on",
            "speed": speed,
        },
        timeout=120,
    )


def generate_clips(dialogues, speaker_refs, ref_cache):
    """Generate English TTS clips with emotion tags and consistent voice refs."""
    print("\n=== Step 5: Generating English speech (Fish Speech 1.5) ===")
    clips_dir = os.path.join(WORKDIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    failures_path = os.path.join(WORKDIR, "failures.txt")

    fallback_ref_b64 = next(iter(ref_cache.values())) if ref_cache else None
    manifest = []
    emotions = {}
    failures = []

    for i, d in enumerate(dialogues):
        clip_path = os.path.join(clips_dir, f"line_{i:04d}.wav")

        # Use pre-classified emotion (set by classify_emotions_bulk in main)
        emotion = d.get("emotion", "neutral")
        speed_factor = d.get("speed_factor", 1.0)
        volume_factor = d.get("volume_factor", 1.0)
        emotions[emotion] = emotions.get(emotion, 0) + 1

        # Skip if already generated (resume support) — verify WAV is readable
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            try:
                with wave.open(clip_path, "rb") as _wf:
                    if _wf.getnframes() == 0:
                        raise ValueError("empty WAV")
            except Exception:
                os.remove(clip_path)  # corrupted — regenerate
            manifest.append({
                "path": clip_path,
                "start": d["start"],
                "target_duration": d["duration"],
                "emotion": emotion,
                "speed_factor": speed_factor,
                "volume_factor": volume_factor,
            })
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(dialogues)} (cached)")
            continue

        if d["duration"] < 0.3:
            continue

        # Use pre-cached base64 reference for this speaker
        ref_b64 = ref_cache.get(d["speaker"], fallback_ref_b64)
        if not ref_b64:
            continue

        # Add emotion tag to text
        tagged_text = add_emotion_tags(d["text"], emotion)
        word_count = len(d["text"].split())
        # Max expected duration: 2.5x subtitle duration or generous word-time estimate
        max_expected = max(d["duration"] * 2.5, word_count * 0.8)

        # Pre-predict English duration — English averages ~2.8 words/sec
        # If estimate exceeds subtitle window, hint Fish Speech to speak faster
        est_duration = word_count / 2.8
        tts_speed = 1.0
        if est_duration > d["duration"] * 1.15 and d["duration"] > 0.5:
            tts_speed = min(est_duration / d["duration"], 1.4)  # cap at 1.4x

        # Retry schedule: (temperature, repetition_penalty)
        retry_schedule = [
            (TTS_TEMPERATURE,       TTS_REPETITION_PENALTY),
            (TTS_TEMPERATURE + 0.1, TTS_REPETITION_PENALTY + 0.3),  # retry 1: more diverse + stricter penalty
            (TTS_TEMPERATURE - 0.1, TTS_REPETITION_PENALTY + 0.5),  # retry 2: more focused + even stricter
        ]

        success = False
        for attempt, (temp, rep_pen) in enumerate(retry_schedule):
            import time
            if attempt > 0:
                time.sleep(attempt)  # 1s, 2s backoff
                if os.path.exists(clip_path):
                    os.remove(clip_path)

            try:
                resp = _tts_request(tagged_text, ref_b64, temp, rep_pen, speed=tts_speed)

                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(clip_path, "wb") as f:
                        f.write(resp.content)

                    # Repetition loop detection
                    clip_dur = _wav_duration(clip_path)
                    if clip_dur > max_expected:
                        safe = d["text"][:40]
                        print(f"  LOOP detected line {i} ({safe}): {clip_dur:.1f}s > {max_expected:.1f}s limit, attempt {attempt+1}")
                        os.remove(clip_path)
                        continue  # try next retry params

                    success = True
                    manifest.append({
                        "path": clip_path,
                        "start": d["start"],
                        "target_duration": d["duration"],
                        "emotion": emotion,
                        "speed_factor": speed_factor,
                        "volume_factor": volume_factor,
                    })
                    break
                else:
                    safe = d["text"][:40]
                    print(f"  WARNING: line {i} ({safe}): status={resp.status_code} size={len(resp.content)} (attempt {attempt+1})")

            except Exception as e:
                safe = d["text"][:40]
                print(f"  WARNING: Failed line {i} ({safe}): {e} (attempt {attempt+1})")

        if not success:
            safe = d["text"][:60]
            failures.append(f"line {i:04d} [{d['start']:.2f}s]: {safe}")

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(dialogues)} | emotions: {emotions} | failures: {len(failures)}")

    # Write failures log
    if failures:
        with open(failures_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failures) + "\n")
        print(f"  FAILED {len(failures)}/{len(dialogues)} lines — see {failures_path}")
    else:
        print(f"  All lines generated successfully")

    print(f"  Generated {len(manifest)} clips | emotions: {emotions}")
    return manifest


# ── Post-process clips (emotion-aware pitch/volume/speed) ───────────────────
def postprocess_clips(manifest):
    """Apply emotion-aware audio effects to each clip using FFmpeg.

    Exclaim: boost volume + slight pitch up
    Whisper: lower volume + low-pass filter (breathy)
    Moaning: slow down slightly + volume boost
    Intense: slight volume boost
    Neutral/question: no change
    """
    print("\n=== Step 5b: Emotion post-processing ===")
    processed = 0

    for m in manifest:
        emotion = m.get("emotion", "neutral")
        if emotion in ("neutral", "question", "laughing", "surprised"):
            continue

        clip_path = m["path"]
        tmp_path = clip_path + ".tmp.wav"

        af_filters = []
        if emotion == "exclaim":
            af_filters.append("volume=1.3")
            af_filters.append("asetrate=44100*1.03,aresample=44100")
        elif emotion == "whisper":
            af_filters.append("volume=0.75")
            af_filters.append("lowpass=f=6000")
        elif emotion == "moaning":
            af_filters.append("volume=1.2")
            af_filters.append("atempo=0.95")
        elif emotion == "intense":
            af_filters.append("volume=1.2")

        if af_filters:
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip_path,
                    "-af", ",".join(af_filters),
                    tmp_path
                ], check=True, capture_output=True)
                os.replace(tmp_path, clip_path)
                processed += 1
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    print(f"  Post-processed {processed} clips")


# ── Assemble Voice Track ───────────────────────────────────────────────────
def assemble_voice_track(manifest, total_duration):
    """Place all clips on timeline with per-clip volume from emotion classification."""
    print("\n=== Step 6: Assembling English voice track ===")

    total_samples = int(total_duration * SAMPLE_RATE)
    timeline = array.array('d', [0.0]) * total_samples

    for m in manifest:
        vol = m.get("volume_factor", 1.0)

        try:
            with wave.open(m["path"], "rb") as wf:
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                fr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
        except Exception:
            continue

        if sw == 2:
            samples = struct.unpack(f"<{len(raw)//2}h", raw)
        else:
            continue

        if nch == 2:
            samples = [(samples[j] + samples[j+1]) / 2 for j in range(0, len(samples), 2)]

        # Resample if needed
        if fr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / fr
            new_len = int(len(samples) * ratio)
            samples = [samples[min(int(j / ratio), len(samples)-1)] for j in range(new_len)]

        # Time-stretch if clip is too long for subtitle window
        clip_dur = len(samples) / SAMPLE_RATE
        target_dur = m["target_duration"]
        if clip_dur > target_dur * 1.1 and target_dur > 0.3:
            ratio = clip_dur / target_dur
            new_len = int(len(samples) / ratio)
            samples = [samples[min(int(j * ratio), len(samples)-1)] for j in range(new_len)]

        # Place on timeline with emotion volume
        start_sample = int(m["start"] * SAMPLE_RATE)
        for j, s in enumerate(samples):
            idx = start_sample + j
            if 0 <= idx < total_samples:
                timeline[idx] += s * vol

    # Normalize
    peak = max(abs(v) for v in timeline) or 1.0
    scale = 32000.0 / peak

    voice_path = os.path.join(WORKDIR, "english_voice.wav")
    with wave.open(voice_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(timeline)}h",
            *[max(-32768, min(32767, int(v * scale))) for v in timeline]))

    print(f"  Voice track: {voice_path} ({total_duration:.1f}s)")
    return voice_path


# ── Mix Final Audio (smart ducking) ────────────────────────────────────────
def mix_audio(voice_path, vocals_path, bg_path, manifest, output_path):
    """Mix English voice + ducked Japanese vocals + background.

    Smart ducking: Japanese vocals are loud between dialogue (moaning, breathing)
    but ducked during English speech. Uses fade transitions (not hard cuts).
    Background music always present at moderate level.
    """
    print("\n=== Step 7: Mixing final audio ===")

    duck_points = [(m["start"], m["start"] + m["target_duration"]) for m in manifest]

    if duck_points:
        # Build volume expression with smooth transitions
        # During English speech: duck to 0.08 (barely audible)
        # Between dialogue: 0.7 (moaning, breathing clearly audible)
        parts = [f"between(t,{s:.2f},{e:.2f})" for s, e in duck_points]
        duck_expr = "+".join(parts)
        # Smooth with a low-pass on the volume envelope
        vocal_filter = (
            f"volume='if({duck_expr},0.08,0.7)':eval=frame,"
            f"afade=t=in:st=0:d=0.5"  # gentle fade in at start
        )
    else:
        vocal_filter = "volume=0.7"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex",
        # English voice: prominent, stereo
        f"[0]aformat=sample_rates=44100:channel_layouts=stereo,volume=2.0[en];"
        # Japanese vocals: ducked during speech, loud between
        f"[1]{vocal_filter}[ja];"
        # Background: always present, moderate
        f"[2]volume=0.8[bg];"
        # Mix all three without normalization
        f"[en][ja][bg]amix=inputs=3:duration=longest:normalize=0[out]",
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final dub: {output_path} ({size_mb:.1f} MB)")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fish Speech 1.5 Local Dubbing (v2)")
    parser.add_argument("--src", default=None, help="Source MKV (auto-detects 1080p)")
    parser.add_argument("--subs", default="subtitle.ass")
    parser.add_argument("--output", default=os.path.join("out", "dub_work", "english_dub.wav"))
    parser.add_argument("--no-cache", action="store_true", help="Force regenerate all clips")
    parser.add_argument("--llm-speakers", action="store_true", help="Use Claude API to identify character names/genders")
    args = parser.parse_args()

    os.makedirs(WORKDIR, exist_ok=True)

    # Check Fish Speech API
    print("Checking Fish Speech API server...")
    try:
        requests.get(f"{API_URL}/v1/models", timeout=5)
        print("  API server: OK")
    except:
        print("  ERROR: Fish Speech API not running!")
        print("  Start: cd fish-speech-v15 && py -3.11 tools/api_server.py \\")
        print("    --llama-checkpoint-path ../models/fish-speech-1.5 \\")
        print("    --decoder-checkpoint-path ../models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \\")
        print("    --half --listen 127.0.0.1:8080")
        sys.exit(1)

    # Auto-detect source MKV
    src = args.src or dub_common.find_mkv_1080()
    if not src:
        print("  ERROR: No 1080p MKV found! Use --src")
        sys.exit(1)
    print(f"Source: {src}")

    # Probe duration
    total_duration = dub_common.probe_duration(src)
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Clear cache if requested
    if args.no_cache:
        import shutil
        clips_dir = os.path.join(WORKDIR, "clips")
        if os.path.exists(clips_dir):
            shutil.rmtree(clips_dir)
        print("  Cleared clip cache")

    # Step 1: Parse subtitles
    print("\n=== Step 1: Parse subtitles ===")
    dialogues = dub_common.parse_ass(args.subs)

    # Step 2: Assign speakers (manual > LLM > ASS styles > auto-detect)
    print("\n=== Step 2: Assign speakers ===")
    labels_path = "speaker_labels.txt" if os.path.isfile("speaker_labels.txt") else None
    speaker_map = None
    if not labels_path and args.llm_speakers:
        speaker_map = dub_common.llm_label_speakers(dialogues, WORKDIR)
    dialogues = dub_common.assign_speakers(dialogues, labels_path=labels_path, speaker_map=speaker_map)

    # Step 2b: Classify emotions with full context window
    print("\n=== Step 2b: Emotion classification (context-aware) ===")
    emotion_dist = dub_common.classify_emotions_bulk(dialogues)
    print(f"  Distribution: {emotion_dist}")

    # Step 3: Separate audio
    vocals, bg = dub_common.separate_audio(src, WORKDIR)

    # Step 4: Extract CALM references per speaker
    speaker_refs = extract_references(dialogues, vocals)

    # Step 4b: Bootstrap English voice refs from Japanese vocals
    ref_cache = bootstrap_english_refs(dialogues, speaker_refs)

    # Step 5: Generate TTS clips
    manifest = generate_clips(dialogues, speaker_refs, ref_cache)

    # Step 5b: Post-process with emotion effects
    dub_common.postprocess_clips(manifest)

    # Step 5c: Time-stretch + normalize volume per clip
    dub_common.time_stretch_clips(manifest)
    dub_common.normalize_clips(manifest)

    # Step 5d: Resolve overlapping dialogue lines
    dub_common.resolve_overlaps(manifest)

    # Step 6: Assemble voice track
    voice_path = dub_common.assemble_voice_track(manifest, total_duration, WORKDIR)

    # Step 7: Mix with smart ducking
    dub_common.mix_audio(voice_path, vocals, bg, manifest, args.output)

    # Step 8: QA report + listening samples
    dub_common.generate_qa_report(manifest, dialogues, WORKDIR, total_duration)
    dub_common.export_listen_samples(args.output, WORKDIR)

    print(f"\n=== DONE! ===")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
