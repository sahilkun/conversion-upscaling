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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(name):
    """Sanitize a speaker name for use as a filename on Windows.

    Removes chars illegal on Windows (<>:"/\\|?* and control chars),
    strips trailing spaces/dots, and truncates to 64 chars.
    """
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    safe = safe.rstrip(' .')
    return (safe[:64] or "speaker")


def _ffmpeg_run(args, **kwargs):
    """Run FFmpeg, re-raising CalledProcessError with readable stderr.

    Without this, all FFmpeg errors produce an unhelpful 'returned non-zero
    exit status 1' with no indication of what went wrong.
    """
    try:
        return subprocess.run(args, check=True, capture_output=True, **kwargs)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace").strip()
        last_lines = "\n".join(stderr.splitlines()[-12:])
        raise RuntimeError(
            f"FFmpeg failed (exit {e.returncode}):\n{last_lines}"
        ) from e


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

# ASS styles that are formatting/layout — NOT character names
_GENERIC_STYLES = {
    "default", "main", "dialogue", "sign", "signs", "note", "notes",
    "op", "ed", "italics", "italic", "title", "comment", "caption",
    "thoughts", "internal", "narrator", "scene", "overlap", "flashback",
    "insert", "on-screen", "onscreen", "screen", "text",
}


def _is_character_style(style_name):
    """Return True if style name looks like a character name (not a formatting style)."""
    return style_name.lower() not in _GENERIC_STYLES and len(style_name) >= 2


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
    """Parse speaker_labels.txt. Returns {line_index: speaker_id}."""
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


def llm_label_speakers(dialogues, workdir, api_key=None):
    """Use Claude API to identify characters and their genders from ASS dialogue.

    Sends unique styles + sample lines to Claude, returns {style_lower: {name, gender}}.
    Result cached to speaker_map_auto.json — subsequent calls reuse it.

    Requires ANTHROPIC_API_KEY env var or api_key param.
    Returns empty dict on failure (pipeline falls back to auto-detect).
    """
    cache_path = os.path.join(workdir, "speaker_map_auto.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  LLM speaker map loaded from cache ({len(cached)} styles)")
        return cached

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("  LLM labeling skipped — no ANTHROPIC_API_KEY")
        return {}

    try:
        import urllib.request
        import urllib.error

        # Build style → sample lines mapping
        style_samples = {}
        for d in dialogues:
            s = d["style"]
            if s not in style_samples:
                style_samples[s] = []
            if len(style_samples[s]) < 4:
                style_samples[s].append(d["text"][:80])

        # Only send character-looking styles
        style_samples = {s: v for s, v in style_samples.items() if _is_character_style(s)}
        if not style_samples:
            return {}

        lines = []
        for style, samples in style_samples.items():
            lines.append(f'Style "{style}":')
            for sample in samples:
                lines.append(f'  - {sample}')
        prompt_body = "\n".join(lines)

        prompt = (
            "You are analyzing anime subtitles. Below are ASS subtitle style names and sample "
            "dialogue lines for each. Identify which styles represent speaking characters "
            "(not sound effects or narration). For each character style, provide:\n"
            "- name: the character's name (use the style name if unclear)\n"
            "- gender: 'female' or 'male'\n\n"
            "Respond with ONLY a JSON object mapping style name (lowercase) to "
            "{\"name\": \"...\", \"gender\": \"...\"}. No explanation.\n\n"
            f"{prompt_body}"
        )

        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        text = result["content"][0]["text"].strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        speaker_map = json.loads(text)
        # Normalize keys to lowercase
        speaker_map = {k.lower(): v for k, v in speaker_map.items()}

        # Validate schema: each value must be a dict with name + gender
        valid = {}
        for style, entry in speaker_map.items():
            if not isinstance(entry, dict):
                print(f"  LLM map: skipping '{style}' — value is not a dict ({entry!r})")
                continue
            name = entry.get("name", "").strip()
            gender = entry.get("gender", "").strip().lower()
            if not name:
                entry["name"] = style  # fall back to style name
            if gender not in ("male", "female"):
                entry["gender"] = "female"  # safe default
            valid[style] = entry
        speaker_map = valid

        if not speaker_map:
            print("  LLM map returned no valid entries, falling back to auto-detect")
            return {}

        os.makedirs(workdir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(speaker_map, f, indent=2)

        print(f"  LLM speaker map: {speaker_map}")
        return speaker_map

    except Exception as e:
        print(f"  LLM labeling failed ({e}), falling back to auto-detect")
        return {}


def assign_speakers(dialogues, labels_path=None, speaker_map=None):
    """Assign speaker ID to each dialogue line.

    Priority:
    1. Manual labels file (speaker_labels.txt)
    2. LLM speaker_map {style_lower: {name, gender}}
    3. ASS style name — if it looks like a character name, use it directly
    4. Keyword gender detection + neighbor propagation fallback

    Each unique character gets their own speaker ID → own voice reference.
    """
    if labels_path and os.path.isfile(labels_path):
        print(f"  Using manual speaker labels: {labels_path}")
        labels = load_speaker_labels(labels_path)
        for label_idx, speaker in labels.items():
            if label_idx < len(dialogues):
                dialogues[label_idx]["speaker"] = speaker
                dialogues[label_idx]["gender"] = speaker
        for d in dialogues:
            if "speaker" not in d:
                # Fallback: use ASS style name if it looks like a character,
                # otherwise use keyword gender detection rather than hard-coding female.
                if _is_character_style(d.get("style", "")):
                    d["speaker"] = d["style"]
                else:
                    gender = detect_gender(d["text"]) or "female"
                    d["speaker"] = gender
                d["gender"] = detect_gender(d["text"]) or "female"

    elif speaker_map:
        print(f"  Using LLM speaker map ({len(speaker_map)} character styles)...")
        for d in dialogues:
            style_key = d["style"].lower()
            if style_key in speaker_map:
                entry = speaker_map[style_key]
                d["speaker"] = entry.get("name", d["style"])
                d["gender"] = entry.get("gender", "female")
            elif _is_character_style(d["style"]):
                # Style not in LLM map but looks like a name — use it directly
                d["speaker"] = d["style"]
                d["gender"] = detect_gender(d["text"]) or "female"
            else:
                d["speaker"] = "female"
                d["gender"] = "female"

    else:
        # Auto-detect: use ASS style names when they look like character names
        print("  Auto-detecting speakers (ASS styles + keyword gender)...")

        # Pass 1: assign speaker from style name where possible
        for d in dialogues:
            if _is_character_style(d["style"]):
                d["speaker"] = d["style"]
            # gender still TBD

        # Pass 2: keyword gender detection
        for d in dialogues:
            d["gender"] = detect_gender(d["text"])

        # Pass 3: propagate gender within same style from neighbors
        for i, d in enumerate(dialogues):
            if d["gender"]:
                continue
            for j in range(max(0, i - 5), min(len(dialogues), i + 5)):
                if dialogues[j]["style"] == d["style"] and dialogues[j].get("gender"):
                    d["gender"] = dialogues[j]["gender"]
                    break
            if not d["gender"]:
                d["gender"] = "female"

        # Pass 4: for generic styles, assign speaker as gender-based ID
        gender_counters = {"female": 0, "male": 0}
        generic_map = {}
        for d in dialogues:
            if "speaker" not in d:
                key = (d["style"], d["gender"])
                if key not in generic_map:
                    gender_counters[d["gender"]] += 1
                    generic_map[key] = f"{d['gender']}{gender_counters[d['gender']]}"
                d["speaker"] = generic_map[key]

    # Ensure gender is set for all lines (needed for voice ref fallback)
    for d in dialogues:
        if "gender" not in d:
            d["gender"] = detect_gender(d["text"]) or "female"

    # Print summary
    speakers = {}
    for d in dialogues:
        speakers.setdefault(d["speaker"], 0)
        speakers[d["speaker"]] += 1
    print(f"  Speakers detected: {len(speakers)}")
    for spk, count in sorted(speakers.items(), key=lambda x: -x[1]):
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
    result = subprocess.run([
        sys.executable, "-m", "demucs", "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", separated_dir, full_audio
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed with exit code {result.returncode}. "
                           "Check that demucs is installed: pip install demucs")

    src_vocals = os.path.join(separated_dir, "htdemucs", "full_audio", "vocals.wav")
    src_bg = os.path.join(separated_dir, "htdemucs", "full_audio", "no_vocals.wav")

    if not os.path.exists(src_vocals):
        raise FileNotFoundError(
            f"Demucs output not found: {src_vocals}\n"
            "Expected path may differ — check separated/ folder structure."
        )
    if not os.path.exists(src_bg):
        raise FileNotFoundError(f"Demucs background output not found: {src_bg}")

    shutil.move(src_vocals, vocals)
    shutil.move(src_bg, bg)

    # Cleanup
    if os.path.exists(full_audio):
        os.remove(full_audio)

    print(f"  Vocals: {vocals}")
    print(f"  Background: {bg}")
    return vocals, bg


# ── Voice Fingerprinting ─────────────────────────────────────────────────────

def _voice_fingerprint(wav_path, max_samples=44100):
    """Compute a simple spectral fingerprint for voice consistency checking.

    Returns a tuple (zcr, low_ratio, high_ratio):
      zcr        — zero crossing rate (proxy for pitch register)
      low_ratio  — energy in 0-1kHz band / total (chest voice indicator)
      high_ratio — energy in 4kHz+ band / total (breathiness indicator)

    Pure stdlib — no numpy required. Uses first 1s of audio.
    Returns None on failure.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            n = min(wf.getnframes(), max_samples)
            raw = wf.readframes(n)

        if sw != 2:
            return None  # not 16-bit PCM — caller used unexpected format

        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        if nch == 2:
            samples = tuple((samples[j] + samples[j+1]) // 2
                            for j in range(0, len(samples), 2))
        if len(samples) < 256:
            return None

        # Zero crossing rate
        crossings = sum(1 for j in range(1, len(samples))
                        if (samples[j-1] >= 0) != (samples[j] >= 0))
        zcr = crossings / len(samples)

        # Frequency band energy via simple band-pass using running sum
        # Approximate: group samples into 20ms frames, compute mean abs amplitude
        # Then use adjacent-frame correlation as low-freq proxy
        frame_sz = max(1, fr // 50)  # 20ms frames
        frame_rms = []
        for start in range(0, len(samples) - frame_sz, frame_sz):
            frame = samples[start:start + frame_sz]
            rms = (sum(s * s for s in frame) / frame_sz) ** 0.5
            frame_rms.append(rms)

        if len(frame_rms) < 4:
            return None

        total_energy = sum(frame_rms) + 1e-9

        # Low-freq proxy: frames that change slowly (adjacent correlation)
        slow_energy = sum(min(frame_rms[j], frame_rms[j+1])
                          for j in range(len(frame_rms) - 1))
        # High-freq proxy: frames with large sudden changes
        fast_energy = sum(abs(frame_rms[j] - frame_rms[j+1])
                          for j in range(len(frame_rms) - 1))

        # Normalize by total_energy only — dividing by len() too would make
        # ratios near-zero for long clips, breaking cross-clip comparison
        low_ratio = slow_energy / total_energy
        high_ratio = fast_energy / total_energy

        return (zcr, low_ratio, high_ratio)

    except Exception:
        return None


def fingerprints_consistent(fp_baseline, fp_new, tolerance=0.35):
    """Return True if fp_new is within tolerance of fp_baseline.

    Checks each dimension independently. Tolerance=0.35 means each component
    can differ by up to 35% relative to the baseline before flagging.
    """
    if fp_baseline is None or fp_new is None:
        return True  # can't check, assume OK
    for b, n in zip(fp_baseline, fp_new):
        if b == 0:
            continue
        if abs(n - b) / abs(b) > tolerance:
            return False
    return True


# ── Calmness Measurement ──────────────────────────────────────────────────────

def _snr_ok(wav_path, min_snr_db=10.0):
    """Quick SNR check — rejects clips with heavy background music bleed.

    Estimates SNR by comparing loudest vs quietest 100ms frames.
    Returns True if SNR >= min_snr_db, False otherwise.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            return True  # can't check, assume OK

        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        if n_channels == 2:
            # Average channels first, then take abs — don't rectify before averaging
            samples = [abs((samples[j] + samples[j+1]) // 2) for j in range(0, len(samples), 2)]
        else:
            samples = [abs(s) for s in samples]

        frame_size = framerate // 10  # 100ms frames
        if frame_size == 0 or len(samples) < frame_size * 3:
            return True  # too short to check

        frame_rms = []
        for i in range(0, len(samples) - frame_size, frame_size):
            frame = samples[i:i + frame_size]
            rms = (sum(s * s for s in frame) / frame_size) ** 0.5
            if rms > 0:
                frame_rms.append(rms)

        if len(frame_rms) < 3:
            return True

        frame_rms.sort()
        noise_floor = sum(frame_rms[:max(1, len(frame_rms) // 10)]) / max(1, len(frame_rms) // 10)
        signal = sum(frame_rms[-(len(frame_rms) // 4):]) / (len(frame_rms) // 4)

        if noise_floor < 1:
            return True  # near-silence, can't compute SNR meaningfully

        import math
        snr = 20 * math.log10(signal / noise_floor)
        return snr >= min_snr_db
    except Exception:
        return True  # on error, don't reject


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

        spk_safe = _safe_filename(spk)
        scored = []
        for idx, (i, d) in enumerate(candidates):
            ref_path = os.path.join(refs_dir, f"{spk_safe}_cand{idx}.wav")
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
                if _snr_ok(ref_path):
                    scored.append((score, ref_path, d["duration"]))
                else:
                    os.remove(ref_path)  # reject BGM-contaminated ref

        if not scored:
            first_path = os.path.join(refs_dir, f"{spk_safe}_cand0.wav")
            final_ref = os.path.join(refs_dir, f"{spk_safe}.wav")
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

        final_ref = os.path.join(refs_dir, f"{spk_safe}.wav")

        if len(chosen) == 1:
            shutil.copy2(chosen[0], final_ref)
        else:
            # Concatenate with silence between clips
            concat_list = os.path.join(refs_dir, f"{spk_safe}_concat.txt")
            silence_path = os.path.join(refs_dir, "silence_0.3s.wav")
            if not os.path.exists(silence_path):
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=mono", "-t", "0.3",
                    "-acodec", "pcm_s16le", silence_path
                ], check=True, capture_output=True)

            with open(concat_list, "w", encoding="utf-8") as f:
                for ci, clip in enumerate(chosen):
                    # FFmpeg concat demuxer requires forward slashes and escaped single quotes
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


def resolve_voice_refs(extracted_refs, voices_dir, training_dir=None, speaker_genders=None):
    """Resolve final voice references per speaker with fallback chain.

    Priority: 1) Extracted calm refs, 2) Series training_data/, 3) Default voices.
    speaker_genders: optional dict {speaker_name: "male"/"female"} from assign_speakers().
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

    def _gender_for(spk):
        """Determine gender for a speaker name.

        Priority: explicit map > name contains 'male'/'female' > default female.
        Named characters (Sachi, Aoi, etc.) default to female since most anime
        protagonists are female; wrong gender is better than wrong voice fallback.
        """
        if speaker_genders and spk in speaker_genders:
            return speaker_genders[spk]
        if spk in ("female", "male"):
            return spk
        if "female" in spk:
            return "female"
        if "male" in spk:
            return "male"
        return "female"  # safe default for named characters

    for spk in speakers:
        wav_path = None
        prompt_text = ""

        # Priority 1: Extracted calm ref
        if extracted_refs and spk in extracted_refs:
            wav_path = extracted_refs[spk]

        # Priority 2: Training data (series-specific)
        if not wav_path and training_dir:
            gender = _gender_for(spk)
            train_dir = os.path.join(training_dir, gender)
            if os.path.isdir(train_dir):
                wavs = sorted([f for f in os.listdir(train_dir) if f.endswith(".wav")])
                if wavs:
                    wav_path = os.path.join(train_dir, wavs[0])

        # Priority 3: Default voice
        if not wav_path:
            gender = _gender_for(spk)
            default = defaults.get(gender)
            if default and os.path.exists(default):
                wav_path = default

        if wav_path:
            result[spk] = (wav_path, prompt_text, [])

    print(f"  Voice refs: {', '.join(f'{s} -> {os.path.basename(p)}' for s, (p, _, _) in result.items())}")
    return result


# ── Emotion Classification ────────────────────────────────────────────────────

# Emotion → (speed_factor, volume_factor)
_EMOTION_PARAMS = {
    "moaning":   (0.95, 1.3),
    "exclaim":   (1.1,  1.3),
    "whisper":   (0.9,  0.7),
    "question":  (1.0,  1.0),
    "intense":   (1.05, 1.2),
    "surprised": (1.0,  1.1),
    "laughing":  (1.0,  1.0),
    "crying":    (0.9,  0.85),
    "angry":     (1.1,  1.35),
    "seductive": (0.88, 0.9),
    "tired":     (0.92, 0.8),
    "playful":   (1.05, 1.05),
    "neutral":   (1.0,  1.0),
}


def _classify_single(text):
    """Single-line emotion classification. Returns emotion string."""
    tl = text.lower().rstrip()

    # Moaning / sexual climax
    if re.search(r"ahh+|ohh+|mmm+|ngh+|haa+|cumming|cum!|shooting out|i.m coming|going to cum", tl):
        return "moaning"
    # Crying / sadness
    if re.search(r"sobbing|sniff|i.m sorry|forgive me|can.t go on|why did|it hurts so much|tears|crying|wept|weep", tl):
        return "crying"
    # Angry
    if re.search(r"how dare|you idiot|shut up|i hate|damn it|get out|never forgive|you bastard|screw you", tl):
        return "angry"
    # Seductive / flirty
    if re.search(r"come here|want you|tease|seduce|just the two|play with|touch me gently|let me show", tl):
        return "seductive"
    # Tired / exhausted
    if re.search(r"so tired|exhausted|can.t anymore|need to rest|sleepy|worn out|i give up", tl):
        return "tired"
    # Playful / teasing
    if re.search(r"gotcha|just kidding|bet you|you wish|teasing|nyah|try and catch|bet you can.t", tl):
        return "playful"
    # Intense arousal
    if re.search(r"can.t stop|so bad|please.*cum|breed me|don.t pull|fill my|deeper|harder|faster|so good|feels so", tl):
        return "intense"
    # Laughing
    if re.search(r"haha|hehe|pfft|tehehe|lol|funny|hilarious", tl):
        return "laughing"
    # Surprised
    if re.search(r"^(huh|what|eh|wait|no way|seriously)\b", tl):
        return "surprised"
    # Whisper — short submissive lines
    if re.search(r"^(yes|okay|huh|please|i see|really)\b", tl) and len(tl) < 20:
        return "whisper"
    # Exclaim — short lines ending in !
    if tl.endswith("!") and len(tl) < 30:
        return "exclaim"
    # Question
    if tl.endswith("?"):
        return "question"

    return "neutral"


def classify_emotion(text, context_before=None, context_after=None):
    """Classify dialogue emotion using a 3-line context window.

    Args:
        text: current dialogue line
        context_before: previous line text (or None)
        context_after: next line text (or None)

    Returns (emotion, speed_factor, volume_factor).

    Context rules:
    - If current is neutral but neighbors are intense/moaning/angry/crying,
      inherit their emotion at reduced intensity (exclaim → neutral stays,
      intense/moaning → intense, angry → exclaim, crying → whisper)
    - Explicit cues in current line always win over context
    """
    emotion = _classify_single(text)

    # Only apply context boost when current line is neutral/question/whisper
    if emotion in ("neutral", "question", "whisper") and (context_before or context_after):
        neighbor_emotions = []
        if context_before:
            neighbor_emotions.append(_classify_single(context_before))
        if context_after:
            neighbor_emotions.append(_classify_single(context_after))

        # Context inheritance map: strong neighbor → softer version for this line
        context_boost = {
            "intense":   "intense",
            "moaning":   "intense",
            "angry":     "exclaim",
            "crying":    "whisper",
            "exclaim":   "exclaim",
            "seductive": "seductive",
        }
        for ne in neighbor_emotions:
            if ne in context_boost:
                emotion = context_boost[ne]
                break

    params = _EMOTION_PARAMS.get(emotion, (1.0, 1.0))
    return emotion, params[0], params[1]


def classify_emotions_bulk(dialogues):
    """Classify emotions for all dialogue lines using the full context window.

    More accurate than per-line calls — each line sees its neighbors.
    Updates dialogues in-place, returns emotion distribution dict.
    """
    texts = [d["text"] for d in dialogues]
    dist = {}
    for i, d in enumerate(dialogues):
        prev_text = texts[i - 1] if i > 0 else None
        next_text = texts[i + 1] if i < len(texts) - 1 else None
        emotion, speed, vol = classify_emotion(d["text"], prev_text, next_text)
        d["emotion"] = emotion
        d["speed_factor"] = speed
        d["volume_factor"] = vol
        dist[emotion] = dist.get(emotion, 0) + 1
    return dist


# ── Post-Processing ───────────────────────────────────────────────────────────

def postprocess_clips(manifest):
    """Apply emotion-aware FFmpeg filters to clips. Returns count processed.

    Emotions with no filter (pass-through): neutral, question, laughing, surprised, playful
    All others get specific audio shaping.
    """
    print("\n=== Emotion post-processing ===")
    processed = 0
    _SKIP = {"neutral", "question", "laughing", "surprised", "playful"}

    for m in manifest:
        emotion = m.get("emotion", "neutral")
        if emotion in _SKIP:
            continue

        clip_path = m["path"]
        tmp_path = clip_path + ".tmp.wav"

        af_filters = []
        if emotion == "exclaim":
            # Louder + slight pitch up for energy
            af_filters += ["volume=1.3", "asetrate=44100*1.03,aresample=44100"]
        elif emotion == "angry":
            # Louder + slight pitch up + subtle distortion edge via treble boost
            af_filters += ["volume=1.35", "asetrate=44100*1.04,aresample=44100",
                           "equalizer=f=3000:width_type=o:width=2:g=3"]
        elif emotion == "whisper":
            # Quieter + low-pass (breathy, muffled)
            af_filters += ["volume=0.75", "lowpass=f=6000"]
        elif emotion == "crying":
            # Quieter + slight tremolo (wavering voice) + low-pass
            af_filters += ["volume=0.85", "tremolo=f=4:d=0.15", "lowpass=f=7000"]
        elif emotion == "moaning":
            # Slightly slower + louder
            af_filters += ["volume=1.2", "atempo=0.95"]
        elif emotion == "seductive":
            # Quieter + slight slow-down + low-pass for breathy quality
            af_filters += ["volume=0.9", "atempo=0.93", "lowpass=f=8000"]
        elif emotion == "tired":
            # Quieter + slower + slight low-pass
            af_filters += ["volume=0.8", "atempo=0.92", "lowpass=f=7500"]
        elif emotion == "intense":
            # Louder only (speed handled by TTS speed_factor)
            af_filters += ["volume=1.2"]

        if af_filters:
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip_path,
                    "-af", ",".join(af_filters),
                    tmp_path
                ], check=True, capture_output=True)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 500:
                    os.replace(tmp_path, clip_path)
                    processed += 1
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    print(f"  Post-processed {processed} clips")
    return processed


def normalize_clips(manifest):
    """Normalize each clip to -20 LUFS for consistent per-line volume.

    Runs ffmpeg loudnorm on every clip in-place. Skips clips that fail.
    Skips clips shorter than 0.5s — loudnorm needs ≥3s analysis window but
    produces acceptable results down to ~0.5s; below that, use peak norm instead.
    Call after time_stretch_clips, before assemble_voice_track.
    """
    print("\n=== Normalizing clip volumes (loudnorm -20 LUFS) ===")
    normalized = 0
    peak_normalized = 0

    for m in manifest:
        clip = m["path"]
        tmp = clip + ".norm.wav"

        try:
            dur = probe_duration(clip)
        except Exception:
            continue

        try:
            if dur >= 0.5:
                # loudnorm for normal-length clips
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip,
                    "-af", "loudnorm=I=-20:TP=-2:LRA=7",
                    "-ar", "44100", tmp
                ], check=True, capture_output=True)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    os.replace(tmp, clip)
                    normalized += 1
            else:
                # Peak normalization for very short clips (<0.5s)
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip,
                    "-af", "dynaudnorm=p=0.95",
                    "-ar", "44100", tmp
                ], check=True, capture_output=True)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    os.replace(tmp, clip)
                    peak_normalized += 1
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"  Normalized {normalized} clips (loudnorm) + {peak_normalized} (peak)")
    return normalized + peak_normalized


def _rubberband_available():
    """Check if FFmpeg was built with the rubberband filter."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True
        )
        return "rubberband" in result.stdout
    except Exception:
        return False


# Cache rubberband availability — checked once per process
_RUBBERBAND = None


def _build_stretch_filter(ratio):
    """Return the best FFmpeg -af filter string for the given tempo ratio.

    Strategy:
      ratio <= 1.3  → atempo (fast, imperceptible quality diff at low ratios)
      ratio 1.3–4.0 → rubberband if available (much better at high ratios),
                       else chained atempo
      ratio > 4.0   → capped at 4.0 with warning
    """
    global _RUBBERBAND
    if _RUBBERBAND is None:
        _RUBBERBAND = _rubberband_available()

    ratio = min(ratio, 4.0)

    if ratio <= 1.3:
        return f"atempo={ratio:.4f}", "atempo"

    if _RUBBERBAND:
        # rubberband: tempo=ratio, pitch=1.0 (preserve pitch)
        return f"rubberband=tempo={ratio:.4f}:pitch=1.0", "rubberband"

    # Fallback: chained atempo (max 2.0 per stage)
    if ratio <= 2.0:
        return f"atempo={ratio:.4f}", "atempo"
    r1 = ratio ** 0.5
    return f"atempo={r1:.4f},atempo={r1:.4f}", "atempo-chain"


def time_stretch_clips(manifest):
    """Pitch-preserving time-stretch for clips exceeding target duration.

    Uses rubberband filter (high quality) when available, falls back to
    chained atempo. rubberband handles ratios up to 4x without artifacts.
    """
    global _RUBBERBAND
    if _RUBBERBAND is None:
        _RUBBERBAND = _rubberband_available()

    print(f"\n=== Time-stretching long clips (engine: {'rubberband' if _RUBBERBAND else 'atempo'}) ===")
    stretched = 0
    methods = {}

    for m in manifest:
        clip = m["path"]
        try:
            dur = probe_duration(clip)
        except Exception:
            continue

        target = m["target_duration"]
        if dur > target * 1.1 and target > 0.3:
            ratio = dur / target
            if ratio > 4.0:
                print(f"  WARNING: ratio {ratio:.2f}x capped at 4.0x for {os.path.basename(clip)}")
            af, method = _build_stretch_filter(ratio)
            methods[method] = methods.get(method, 0) + 1

            tmp = clip + ".tempo.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip, "-af", af, tmp],
                    check=True, capture_output=True
                )
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    os.replace(tmp, clip)
                    stretched += 1
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)

    method_summary = ", ".join(f"{m}:{n}" for m, n in methods.items()) if methods else "none"
    print(f"  Stretched {stretched} clips ({method_summary})")
    return stretched


# ── Overlap Resolution ───────────────────────────────────────────────────────

def resolve_overlaps(manifest):
    """Detect and resolve overlapping TTS clips on the timeline.

    Strategy (in order of severity):
      - overlap < 200ms: trim end of earlier clip
      - overlap 200–500ms: speed up earlier clip to fit
      - overlap > 500ms: log warning, leave both (amix handles it)

    Modifies manifest in-place (updates target_duration of affected entries).
    """
    if len(manifest) < 2:
        return

    resolved = 0
    warned_set = set()

    # Multi-pass: repeat until no more overlaps can be fixed (max 5 passes).
    # A single forward pass misses cascade overlaps (A overlaps B overlaps C —
    # fixing A→B may introduce or reveal a new B→C overlap in the same pass).
    for _pass in range(5):
        changes_this_pass = 0

        for i in range(len(manifest) - 1):
            cur = manifest[i]
            nxt = manifest[i + 1]

            cur_end = cur["start"] + cur["target_duration"]
            overlap = cur_end - nxt["start"]

            if overlap <= 0:
                continue

            if overlap < 0.2:
                # Trim: shorten current clip's target duration
                manifest[i]["target_duration"] = nxt["start"] - cur["start"]
                resolved += 1
                changes_this_pass += 1

            elif overlap < 0.5:
                # Speed up: re-stretch current clip to fit exactly before next
                new_target = nxt["start"] - cur["start"]
                if new_target > 0.1:
                    clip = cur["path"]
                    try:
                        dur = probe_duration(clip)
                        ratio = dur / new_target
                        ratio = min(ratio, 4.0)
                        if ratio <= 2.0:
                            af = f"atempo={ratio:.4f}"
                        else:
                            r1 = ratio ** 0.5
                            af = f"atempo={r1:.4f},atempo={r1:.4f}"
                        tmp = clip + ".overlap.wav"
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", clip, "-af", af, tmp],
                            check=True, capture_output=True
                        )
                        if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                            os.replace(tmp, clip)
                            manifest[i]["target_duration"] = new_target
                            resolved += 1
                            changes_this_pass += 1
                    except Exception:
                        pass
            elif i not in warned_set:
                print(f"  OVERLAP WARNING: line {i} overlaps next by {overlap*1000:.0f}ms — leaving as-is")
                warned_set.add(i)

        if changes_this_pass == 0:
            break  # converged — no more fixable overlaps

    if resolved or warned_set:
        print(f"  Overlap resolution: {resolved} fixed, {len(warned_set)} warnings")


# ── Timeline Assembly ─────────────────────────────────────────────────────────

def assemble_voice_track(manifest, total_duration, workdir, sample_rate=44100):
    """Place all clips on timeline using FFmpeg adelay+amix. Returns output path.

    Replaces the old pure-Python sample loop (was O(N*samples) — ~40M iterations
    for a 24-min episode, ~20-40 min assembly time). FFmpeg handles all format
    conversion, resampling, delay, and mixing in native C.

    Uses -filter_complex_script to write the filter graph to a temp file, which
    avoids Windows command-line length limits (32K chars) for 300+ clip episodes.
    """
    print("\n=== Assembling English voice track ===")

    manifest = sorted(manifest, key=lambda m: m["start"])
    voice_path = os.path.join(workdir, "english_voice.wav")

    if not manifest:
        _ffmpeg_run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-t", f"{total_duration:.3f}",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-acodec", "pcm_s16le", voice_path,
        ])
        print(f"  Voice track (empty): {voice_path}")
        return voice_path

    # Filter graph:
    #   Input 0 — silent base at exactly total_duration (sets output length)
    #   Inputs 1..N — clips, each aformat'd to mono/target-rate, adelay'd to
    #                 their subtitle start position, volume-adjusted
    #   amix all together, duration=first so output = total_duration
    script_path = os.path.join(workdir, "voice_filter.txt")
    args = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-t", f"{total_duration:.3f}",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
    ]

    filter_lines = []
    for k, m in enumerate(manifest):
        args += ["-i", m["path"]]
        delay_ms = int(m["start"] * 1000)
        vol = m.get("volume_factor", 1.0)
        inp = k + 1  # input 0 is the silence base
        filter_lines.append(
            f"[{inp}]aformat=sample_rates={sample_rate}:channel_layouts=mono,"
            f"adelay={delay_ms},volume={vol:.4f}[d{k}]"
        )

    base_and_clips = "[0]" + "".join(f"[d{k}]" for k in range(len(manifest)))
    filter_lines.append(
        f"{base_and_clips}amix=inputs={len(manifest) + 1}:duration=first:normalize=0[out]"
    )

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(";\n".join(filter_lines))

    args += [
        "-filter_complex_script", script_path,
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
        voice_path,
    ]

    try:
        _ffmpeg_run(args)
    finally:
        try:
            os.remove(script_path)
        except Exception:
            pass

    # Duration validation
    try:
        actual_dur = probe_duration(voice_path)
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

    Smart ducking via sidechaincompress: the English voice track is used as the
    sidechain key — JP vocals are automatically ducked whenever English speech
    is present. This replaces the O(N) expression string that exceeded FFmpeg's
    parser limit for 300+ line episodes.
    Attack 5ms / release 200ms gives punch-in/out without pumping.
    Voice gets pseudo-stereo widening (haas effect) to avoid flat mono sound.
    Background music always present at moderate level.
    """
    print("\n=== Mixing final audio ===")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # [0] = English voice (mono), [1] = JP vocals, [2] = background
    #
    # 1. Split voice: one copy for haas chain, one as sidechain key for ducking
    # 2. Haas pseudo-stereo on voice (original L, 20ms delayed R)
    # 3. Sidechain-compress JP vocals: duck when English voice has signal
    #    threshold=0.02 (~-34dBFS) triggers on voice, ratio=10 gives ~10x attenuation
    # 4. Final mix + dynamics + loudnorm
    # sidechaincompress requires sidechain channel count to match the main signal.
    # JP vocals from Demucs are stereo; English voice (sidechain) is mono.
    # Explicitly upmix the mono sidechain to stereo to avoid channel layout
    # mismatch crash on FFmpeg < 7 (and deprecation warning on FFmpeg 7+).
    filter_complex = (
        "[0]asplit=2[en_raw][en_sc];"
        "[en_raw]asplit[vl][vr];"
        "[vr]adelay=20[vrd];"
        "[vl][vrd]amerge,volume=2.0[en];"
        "[en_sc]aformat=channel_layouts=stereo[en_sc_st];"
        "[1][en_sc_st]sidechaincompress="
        "threshold=0.02:ratio=10:attack=5:release=200:makeup=1[ja_sc];"
        "[ja_sc]volume=0.7[ja];"
        "[2]volume=0.8[bg];"
        "[en][ja][bg]amix=inputs=3:duration=longest:normalize=0,"
        "compand=attacks=0.05:decays=0.3:points=-80/-80|-30/-15|-10/-8|0/-6:soft-knee=6:gain=4,"
        "loudnorm=I=-16:TP=-1.5:LRA=7[out]"
    )

    _ffmpeg_run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ])

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final dub: {output_path} ({size_mb:.1f} MB)")


# ── QA Report ────────────────────────────────────────────────────────────────

def generate_qa_report(manifest, dialogues, workdir, total_duration):
    """Generate a QA report after dubbing. Writes dub_work/qa_report.txt.

    Covers: generation rate, failed lines, timing accuracy, emotion breakdown,
    speaker breakdown, time-stretched clips, and duration validation.
    """
    import datetime

    report_path = os.path.join(workdir, "qa_report.txt")
    failures_path = os.path.join(workdir, "failures.txt")

    total_lines = len(dialogues)
    generated = len(manifest)
    failed_lines = []
    if os.path.exists(failures_path):
        with open(failures_path, encoding="utf-8") as f:
            failed_lines = [l.strip() for l in f if l.strip()]

    # Emotion distribution
    emotion_dist = {}
    for d in dialogues:
        e = d.get("emotion", "neutral")
        emotion_dist[e] = emotion_dist.get(e, 0) + 1

    # Speaker distribution
    speaker_dist = {}
    for d in dialogues:
        s = d.get("speaker", "unknown")
        speaker_dist[s] = speaker_dist.get(s, 0) + 1

    # Timing accuracy: compare clip duration vs target
    duration_deltas = []
    stretched = 0
    tight_fits = 0  # clips within 10% of target
    for m in manifest:
        try:
            actual = probe_duration(m["path"])
            target = m["target_duration"]
            delta = actual - target
            duration_deltas.append(delta)
            ratio = actual / target if target > 0 else 1.0
            if ratio > 1.1:
                stretched += 1
            if abs(delta) <= target * 0.1:
                tight_fits += 1
        except Exception:
            pass

    avg_delta = sum(duration_deltas) / len(duration_deltas) if duration_deltas else 0.0
    max_over = max((d for d in duration_deltas if d > 0), default=0.0)

    # Total voice coverage (how much of the video has English speech)
    voice_seconds = sum(m["target_duration"] for m in manifest)
    coverage_pct = (voice_seconds / total_duration * 100) if total_duration > 0 else 0.0

    lines = [
        f"Dubbing QA Report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "── Generation ──────────────────────────────────────────────",
        f"  Lines total:      {total_lines}",
        f"  Lines generated:  {generated}  ({generated/total_lines*100:.1f}%)" if total_lines else "  Lines generated:  0",
        f"  Lines failed:     {len(failed_lines)}",
        f"  Voice coverage:   {voice_seconds:.1f}s / {total_duration:.1f}s  ({coverage_pct:.1f}%)",
        "",
        "── Timing Accuracy ─────────────────────────────────────────",
        f"  Avg duration delta:  {avg_delta:+.3f}s  (+ = clip longer than target)",
        f"  Max overrun:         {max_over:.3f}s",
        f"  Tight fits (±10%):   {tight_fits}/{generated}",
        f"  Time-stretched:      {stretched}",
        "",
        "── Emotion Distribution ────────────────────────────────────",
    ]
    for emotion, count in sorted(emotion_dist.items(), key=lambda x: -x[1]):
        pct = count / total_lines * 100 if total_lines else 0
        lines.append(f"  {emotion:<12} {count:>4}  ({pct:.1f}%)")

    lines += [
        "",
        "── Speaker Distribution ────────────────────────────────────",
    ]
    for spk, count in sorted(speaker_dist.items(), key=lambda x: -x[1]):
        pct = count / total_lines * 100 if total_lines else 0
        lines.append(f"  {spk:<20} {count:>4}  ({pct:.1f}%)")

    if failed_lines:
        lines += [
            "",
            "── Failed Lines ────────────────────────────────────────────",
        ]
        for fl in failed_lines:
            lines.append(f"  {fl}")

    lines += ["", "=" * 60]

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n=== QA Report ===")
    print(report_text)
    return report_path


def export_listen_samples(dub_path, workdir, n=5, clip_len=30):
    """Export N evenly-spaced 30s clips from the final dub for quick listening check.

    Skips first and last 5% of the dub (intros/outros tend to be music only).
    Saves to dub_work/samples/sample_01.wav ... sample_N.wav.
    """
    print(f"\n=== Exporting {n} listening samples ({clip_len}s each) ===")
    samples_dir = os.path.join(workdir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    try:
        total = probe_duration(dub_path)
    except Exception as e:
        print(f"  Could not probe dub duration: {e}")
        return

    usable_start = total * 0.05
    usable_end = total * 0.95
    usable = usable_end - usable_start

    if usable < clip_len:
        print(f"  Dub too short for samples ({total:.1f}s)")
        return

    # Evenly space N start points across usable range
    if n == 1:
        starts = [usable_start + usable / 2 - clip_len / 2]
    else:
        step = (usable - clip_len) / (n - 1)
        starts = [usable_start + i * step for i in range(n)]

    exported = 0
    for i, start in enumerate(starts):
        out_path = os.path.join(samples_dir, f"sample_{i+1:02d}.wav")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}", "-t", f"{clip_len}",
                "-i", dub_path,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                out_path
            ], check=True, capture_output=True)
            exported += 1
        except Exception as e:
            print(f"  Sample {i+1} failed: {e}")

    print(f"  Exported {exported}/{n} samples → {samples_dir}")


# ── MKV Auto-Detection ────────────────────────────────────────────────────────

def find_mkv_1080(directory="."):
    """Auto-detect 1080p MKV in directory."""
    for f in os.listdir(directory):
        if f.endswith(".mkv") and "1080" in f.lower():
            return os.path.join(directory, f)
    return None
