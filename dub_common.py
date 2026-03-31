#!/usr/bin/env python3
"""
Shared dubbing utilities — subtitle parsing, speaker detection, audio separation,
voice reference extraction, emotion processing, timeline assembly, and mixing.

All functions take explicit path parameters. Zero hardcoded paths.
"""

import array
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import wave

SAMPLE_RATE = 44100


# ── Subtitle Parsing ─────────────────────────────────────────────────────────

def parse_ass(path):
    """Parse ASS subtitle file into dialogue list.

    Skips sign/credit/OP/ED styles. Strips ASS override tags and line breaks.
    Returns list of {start, end, duration, style, text} sorted by start time.
    """
    dialogues = []
    skip_styles = {"sign", "signs", "credits", "credit", "note", "ed", "op"}

    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue

        style = parts[3].strip()
        if style.lower() in skip_styles:
            continue

        h1, m1, s1 = parts[1].strip().split(":")
        start = int(h1) * 3600 + int(m1) * 60 + float(s1)
        h2, m2, s2 = parts[2].strip().split(":")
        end = int(h2) * 3600 + int(m2) * 60 + float(s2)

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
    print(f"  Parsed {len(dialogues)} dialogue lines from {os.path.basename(path)}")
    return dialogues


# ── Speaker Detection ─────────────────────────────────────────────────────────

def detect_gender(text):
    """Keyword-based gender detection from dialogue text.

    Returns 'female', 'male', or None if undetermined.
    """
    tl = text.lower()
    if re.search(r"pussy|tits|my breasts|my body|cum inside me|breed me|womb|panties", tl):
        return "female"
    if re.search(r"my cock|my dick|balls|i.ll cum|ejaculate|erection", tl):
        return "male"
    return None


def load_speaker_labels(labels_path):
    """Parse speaker_labels.txt. Returns {line_index: 'female'|'male'}."""
    labels = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([FM])(?:,\s*[FM])?\s+(\d+)\s+", line)
            if match:
                speaker = "female" if match.group(1) == "F" else "male"
                idx = int(match.group(2))
                labels[idx] = speaker
    return labels


def assign_speakers(dialogues, labels_path=None):
    """Assign speaker/gender to each dialogue line.

    If labels_path exists and is a file, use manual labels.
    Otherwise, auto-detect using keyword matching + neighbor propagation.
    """
    if labels_path and os.path.isfile(labels_path):
        print(f"  Using manual speaker labels: {labels_path}")
        labels = load_speaker_labels(labels_path)
        labeled_indices = sorted(labels.keys())
        for i, label_idx in enumerate(labeled_indices):
            if i < len(dialogues):
                dialogues[i]["speaker"] = labels[label_idx]
                dialogues[i]["gender"] = labels[label_idx]
        for d in dialogues:
            if "speaker" not in d:
                d["speaker"] = "female"
                d["gender"] = "female"
    else:
        print("  Auto-detecting speaker gender from text...")
        # Step 1: Keyword detection
        for d in dialogues:
            d["gender"] = detect_gender(d["text"])

        # Step 2: Propagate from neighbors within same style
        for i, d in enumerate(dialogues):
            if d["gender"]:
                continue
            for j in range(max(0, i - 5), min(len(dialogues), i + 5)):
                if dialogues[j]["style"] == d["style"] and dialogues[j]["gender"]:
                    d["gender"] = dialogues[j]["gender"]
                    break
            if not d["gender"]:
                d["gender"] = "female"

        # Step 3: Assign speaker IDs per (style, gender) combo
        speaker_map = {}
        counters = {"female": 0, "male": 0}
        for d in dialogues:
            key = (d["style"], d["gender"])
            if key not in speaker_map:
                counters[d["gender"]] += 1
                speaker_map[key] = f"{d['gender']}{counters[d['gender']]}"
            d["speaker"] = speaker_map[key]

    # Print summary
    speakers = {}
    for d in dialogues:
        speakers.setdefault(d["speaker"], 0)
        speakers[d["speaker"]] += 1
    for spk, count in speakers.items():
        print(f"    {spk}: {count} lines")

    return dialogues


# ── Duration Probe ────────────────────────────────────────────────────────────

def probe_duration(path):
    """Get duration in seconds from any media file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# ── Audio Separation (Demucs) ─────────────────────────────────────────────────

def separate_audio(src_mkv, workdir):
    """Run Demucs vocal/background separation. Returns (vocals_path, bg_path).

    Caches results — skips if vocals.wav and background.wav already exist.
    """
    print("\n=== Separating vocals/background (Demucs) ===")

    vocals = os.path.join(workdir, "vocals.wav")
    bg = os.path.join(workdir, "background.wav")

    if os.path.exists(vocals) and os.path.exists(bg):
        print("  Cached, skipping.")
        return vocals, bg

    os.makedirs(workdir, exist_ok=True)

    # Extract full audio from MKV
    full_audio = os.path.join(workdir, "full_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", src_mkv,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        full_audio
    ], check=True, capture_output=True)

    # Run Demucs
    separated_dir = os.path.join(workdir, "separated")
    subprocess.run([
        sys.executable, "-m", "demucs", "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", separated_dir, full_audio
    ], check=True)

    src_vocals = os.path.join(separated_dir, "htdemucs", "full_audio", "vocals.wav")
    src_bg = os.path.join(separated_dir, "htdemucs", "full_audio", "no_vocals.wav")

    shutil.move(src_vocals, vocals)
    shutil.move(src_bg, bg)

    # Cleanup
    if os.path.exists(full_audio):
        os.remove(full_audio)

    print(f"  Vocals: {vocals}")
    print(f"  Background: {bg}")
    return vocals, bg


# ── Calmness Measurement ──────────────────────────────────────────────────────

def measure_calmness(wav_path):
    """Measure vocal calmness via RMS variance. Lower = calmer."""
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


# ── Voice Reference Resolution ────────────────────────────────────────────────

def extract_calm_refs(dialogues, vocals_path, workdir,
                      min_dur=2.0, max_dur=None, max_candidates=12,
                      target_seconds=12.0):
    """Extract calmest vocal clips per speaker. Returns {speaker: wav_path}.

    Concatenates top calm clips into a composite reference for voice stability.
    GPT-SoVITS callers should pass max_dur=10.0 (its limit).
    """
    print("\n=== Extracting calm voice references ===")
    refs_dir = os.path.join(workdir, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    speaker_candidates = {}
    for i, d in enumerate(dialogues):
        dur = d["duration"]
        if dur < min_dur:
            continue
        if max_dur and dur > max_dur:
            continue
        spk = d["speaker"]
        if spk not in speaker_candidates:
            speaker_candidates[spk] = []
        if len(speaker_candidates[spk]) < max_candidates:
            speaker_candidates[spk].append((i, d))

    speaker_refs = {}
    for spk, candidates in speaker_candidates.items():
        print(f"  {spk}: evaluating {len(candidates)} candidates...")

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
            first_path = os.path.join(refs_dir, f"{spk}_cand0.wav")
            final_ref = os.path.join(refs_dir, f"{spk}.wav")
            if os.path.exists(first_path):
                shutil.copy2(first_path, final_ref)
                speaker_refs[spk] = final_ref
            continue

        scored.sort(key=lambda x: x[0])
        chosen = []
        total_dur = 0.0
        for score, path, dur in scored:
            chosen.append(path)
            total_dur += dur
            if total_dur >= target_seconds:
                break

        final_ref = os.path.join(refs_dir, f"{spk}.wav")

        if len(chosen) == 1:
            shutil.copy2(chosen[0], final_ref)
        else:
            # Concatenate with silence between clips
            concat_list = os.path.join(refs_dir, f"{spk}_concat.txt")
            silence_path = os.path.join(refs_dir, "silence_0.3s.wav")
            if not os.path.exists(silence_path):
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=mono", "-t", "0.3",
                    "-acodec", "pcm_s16le", silence_path
                ], check=True, capture_output=True)

            with open(concat_list, "w") as f:
                for ci, clip in enumerate(chosen):
                    f.write(f"file '{os.path.abspath(clip)}'\n")
                    if ci < len(chosen) - 1:
                        f.write(f"file '{os.path.abspath(silence_path)}'\n")

            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                final_ref
            ], check=True, capture_output=True)
            os.remove(concat_list)
            print(f"    -> Composite: {len(chosen)} clips, ~{total_dur:.1f}s")

        speaker_refs[spk] = final_ref

    # Cleanup candidates
    for f in os.listdir(refs_dir):
        if "_cand" in f:
            os.remove(os.path.join(refs_dir, f))
    silence = os.path.join(refs_dir, "silence_0.3s.wav")
    if os.path.exists(silence):
        os.remove(silence)

    return speaker_refs


def resolve_voice_refs(extracted_refs, voices_dir, training_dir=None):
    """Resolve final voice references per speaker with fallback chain.

    Priority: 1) Extracted calm refs, 2) Series training_data/, 3) Default voices.
    Returns {speaker: (wav_path, prompt_text, aux_paths)}.
    """
    result = {}
    defaults = {
        "female": os.path.join(voices_dir, "default_female.wav"),
        "male": os.path.join(voices_dir, "default_male.wav"),
    }

    speakers = set(extracted_refs.keys()) if extracted_refs else set()
    # Ensure at least female and male exist
    speakers.update(["female", "male"])

    for spk in speakers:
        wav_path = None
        prompt_text = ""

        # Priority 1: Extracted calm ref
        if extracted_refs and spk in extracted_refs:
            wav_path = extracted_refs[spk]

        # Priority 2: Training data (series-specific)
        if not wav_path and training_dir:
            gender = "female" if "female" in spk else "male"
            train_dir = os.path.join(training_dir, gender)
            if os.path.isdir(train_dir):
                wavs = sorted([f for f in os.listdir(train_dir) if f.endswith(".wav")])
                if wavs:
                    wav_path = os.path.join(train_dir, wavs[0])

        # Priority 3: Default voice
        if not wav_path:
            gender = "female" if "female" in spk else "male"
            default = defaults.get(gender)
            if default and os.path.exists(default):
                wav_path = default

        if wav_path:
            result[spk] = (wav_path, prompt_text, [])

    print(f"  Voice refs: {', '.join(f'{s} -> {os.path.basename(p)}' for s, (p, _, _) in result.items())}")
    return result


# ── Emotion Classification ────────────────────────────────────────────────────

def classify_emotion(text):
    """Classify dialogue emotion. Returns (emotion, speed_factor, volume_factor)."""
    tl = text.lower().rstrip()

    if re.search(r"ahh|ohh|mmm|ngh|haa|cumming|cum!|shooting out", tl):
        return "moaning", 0.95, 1.3
    if tl.endswith("!") and len(tl) < 25:
        return "exclaim", 1.1, 1.3
    if re.search(r"^(yes|okay|huh|please)\b", tl) and len(tl) < 20:
        return "whisper", 0.9, 0.7
    if tl.endswith("?"):
        return "question", 1.0, 1.0
    if re.search(r"can.t stop|can.t wait|so bad|please.*cum|breed me|don.t pull|fill my|deeper|harder|faster|so good|feels", tl):
        return "intense", 1.05, 1.2
    if re.search(r"^(huh|what|eh|wait)\b", tl):
        return "surprised", 1.0, 1.1
    if re.search(r"haha|hehe|lol|funny", tl):
        return "laughing", 1.0, 1.0
    return "neutral", 1.0, 1.0


# ── Post-Processing ───────────────────────────────────────────────────────────

def postprocess_clips(manifest):
    """Apply emotion-aware FFmpeg filters to clips. Returns count processed."""
    print("\n=== Emotion post-processing ===")
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
    return processed


def time_stretch_clips(manifest):
    """Pitch-preserving time-stretch for clips exceeding target duration."""
    print("\n=== Time-stretching long clips ===")
    stretched = 0

    for m in manifest:
        clip = m["path"]
        try:
            dur = probe_duration(clip)
        except Exception:
            continue

        target = m["target_duration"]
        if dur > target * 1.2 and target > 0.3:
            ratio = min(dur / target, 1.15)
            tmp = clip + ".tempo.wav"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip,
                    "-af", f"atempo={ratio:.3f}",
                    tmp
                ], check=True, capture_output=True)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    os.replace(tmp, clip)
                    stretched += 1
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print(f"  Stretched {stretched} clips (max 1.15x, pitch preserved)")
    return stretched


# ── Timeline Assembly ─────────────────────────────────────────────────────────

def assemble_voice_track(manifest, total_duration, workdir, sample_rate=44100):
    """Place all clips on timeline, normalize, write WAV. Returns output path."""
    print("\n=== Assembling English voice track ===")

    total_samples = int(total_duration * sample_rate)
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

        if sw != 2:
            continue
        samples = struct.unpack(f"<{len(raw)//2}h", raw)

        # Stereo to mono
        if nch == 2:
            samples = [(samples[j] + samples[j+1]) / 2 for j in range(0, len(samples), 2)]

        # Resample if needed
        if fr != sample_rate:
            ratio = sample_rate / fr
            new_len = int(len(samples) * ratio)
            samples = [samples[min(int(j / ratio), len(samples)-1)] for j in range(new_len)]

        # In-memory time-stretch if too long
        clip_dur = len(samples) / sample_rate
        target_dur = m["target_duration"]
        if clip_dur > target_dur * 1.1 and target_dur > 0.3:
            ratio = clip_dur / target_dur
            new_len = int(len(samples) / ratio)
            samples = [samples[min(int(j * ratio), len(samples)-1)] for j in range(new_len)]

        # Place on timeline
        start_sample = int(m["start"] * sample_rate)
        for j, s in enumerate(samples):
            idx = start_sample + j
            if 0 <= idx < total_samples:
                timeline[idx] += s * vol

    # Normalize
    peak = max(abs(v) for v in timeline) or 1.0
    scale = 32000.0 / peak

    voice_path = os.path.join(workdir, "english_voice.wav")
    with wave.open(voice_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(timeline)}h",
            *[max(-32768, min(32767, int(v * scale))) for v in timeline]))

    # Duration validation
    try:
        with wave.open(voice_path, "rb") as _wf:
            actual_dur = _wf.getnframes() / _wf.getframerate()
        delta = abs(actual_dur - total_duration)
        if delta > 2.0:
            print(f"  WARNING: voice track duration mismatch — expected {total_duration:.1f}s, got {actual_dur:.1f}s (delta {delta:.1f}s)")
        else:
            print(f"  Duration check OK: {actual_dur:.1f}s (expected {total_duration:.1f}s)")
    except Exception as _e:
        print(f"  WARNING: could not verify voice track duration: {_e}")

    print(f"  Voice track: {voice_path} ({total_duration:.1f}s)")
    return voice_path


# ── Audio Mixing ──────────────────────────────────────────────────────────────

def mix_audio(voice_path, vocals_path, bg_path, manifest, output_path):
    """Mix English voice + ducked Japanese vocals + background.

    Smart ducking: JP vocals muted during English speech, audible between.
    Background music always present at moderate level.
    """
    print("\n=== Mixing final audio ===")

    duck_points = [(m["start"], m["start"] + m["target_duration"]) for m in manifest]

    if duck_points:
        parts = [f"between(t,{s:.2f},{e:.2f})" for s, e in duck_points]
        duck_expr = "+".join(parts)
        vocal_filter = (
            f"volume='if({duck_expr},0.08,0.7)':eval=frame,"
            f"afade=t=in:st=0:d=0.5"
        )
    else:
        vocal_filter = "volume=0.7"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex",
        f"[0]aformat=sample_rates=44100:channel_layouts=stereo,volume=2.0[en];"
        f"[1]{vocal_filter}[ja];"
        f"[2]volume=0.8[bg];"
        f"[en][ja][bg]amix=inputs=3:duration=longest:normalize=0,"
        f"compand=attacks=0.05:decays=0.3:points=-80/-80|-30/-15|-10/-8|0/-6:soft-knee=6:gain=4,"
        f"loudnorm=I=-16:TP=-1.5:LRA=7[out]",
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final dub: {output_path} ({size_mb:.1f} MB)")


# ── MKV Auto-Detection ────────────────────────────────────────────────────────

def find_mkv_1080(directory="."):
    """Auto-detect 1080p MKV in directory."""
    for f in os.listdir(directory):
        if f.endswith(".mkv") and "1080" in f.lower():
            return os.path.join(directory, f)
    return None
