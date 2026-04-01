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
        labeled_indices = sorted(labels.keys())
        for i, label_idx in enumerate(labeled_indices):
            if i < len(dialogues):
                dialogues[i]["speaker"] = labels[label_idx]
                dialogues[i]["gender"] = labels[label_idx]
        for d in dialogues:
            if "speaker" not in d:
                d["speaker"] = "female"
                d["gender"] = "female"

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
            samples = [abs(samples[j] + samples[j+1]) // 2 for j in range(0, len(samples), 2)]
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
                if _snr_ok(ref_path):
                    scored.append((score, ref_path, d["duration"]))
                else:
                    os.remove(ref_path)  # reject BGM-contaminated ref

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
    Call after time_stretch_clips, before assemble_voice_track.
    """
    print("\n=== Normalizing clip volumes (loudnorm -20 LUFS) ===")
    normalized = 0

    for m in manifest:
        clip = m["path"]
        tmp = clip + ".norm.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", clip,
                "-af", "loudnorm=I=-20:TP=-2:LRA=7",
                "-ar", "44100", tmp
            ], check=True, capture_output=True)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                os.replace(tmp, clip)
                normalized += 1
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"  Normalized {normalized}/{len(manifest)} clips")
    return normalized


def time_stretch_clips(manifest):
    """Pitch-preserving time-stretch for clips exceeding target duration.

    Uses FFmpeg atempo (0.5–2.0x). For ratios >2.0x, chains two atempo filters.
    Removes the 1.15x cap — will stretch up to 2.0x to fit subtitle window.
    """
    print("\n=== Time-stretching long clips ===")
    stretched = 0

    for m in manifest:
        clip = m["path"]
        try:
            dur = probe_duration(clip)
        except Exception:
            continue

        target = m["target_duration"]
        if dur > target * 1.1 and target > 0.3:
            ratio = dur / target
            ratio = min(ratio, 4.0)  # hard cap: never compress more than 4x

            # atempo is capped at 2.0; chain two filters if needed
            if ratio <= 2.0:
                af = f"atempo={ratio:.4f}"
            else:
                # e.g. ratio=3.0 -> atempo=1.732,atempo=1.732 (sqrt(3))
                r1 = ratio ** 0.5
                af = f"atempo={r1:.4f},atempo={r1:.4f}"

            tmp = clip + ".tempo.wav"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", clip, "-af", af, tmp
                ], check=True, capture_output=True)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    os.replace(tmp, clip)
                    stretched += 1
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print(f"  Stretched {stretched} clips (pitch-preserving atempo, up to 4x)")
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
    warned = 0

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
                except Exception:
                    pass
        else:
            print(f"  OVERLAP WARNING: line {i} overlaps next by {overlap*1000:.0f}ms — leaving as-is")
            warned += 1

    if resolved or warned:
        print(f"  Overlap resolution: {resolved} fixed, {warned} warnings")


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

def _build_duck_filter(duck_points, vol_normal=0.7, vol_duck=0.08, fade=0.05):
    """Build an FFmpeg volume expression with smooth 50ms fade in/out ramps.

    Each segment uses a trapezoid envelope:
      rise over [s-fade, s], hold over [s, e], fall over [e, e+fade].
    Expressed as: max(0,min(1,(t-s)/fade+1)) * max(0,min(1,(e-t)/fade+1))
    All segments summed and clamped to [0,1] to handle adjacent lines.
    Final volume: vol_normal - (vol_normal - vol_duck) * duck_factor
    """
    if not duck_points:
        return f"volume={vol_normal}"

    F = fade
    seg_exprs = [
        f"max(0,min(1,(t-{s:.3f})/{F}+1))*max(0,min(1,({e:.3f}-t)/{F}+1))"
        for s, e in duck_points
    ]
    sum_expr = "+".join(seg_exprs)
    duck_range = vol_normal - vol_duck  # 0.62
    return f"volume='{vol_normal}-{duck_range:.4f}*min(1,{sum_expr})':eval=frame"


def mix_audio(voice_path, vocals_path, bg_path, manifest, output_path):
    """Mix English voice + ducked Japanese vocals + background.

    Smart ducking: JP vocals muted during English speech, audible between.
    Smooth 50ms fade in/out ramps on duck transitions.
    Voice gets pseudo-stereo widening (haas effect) to avoid flat mono sound.
    Background music always present at moderate level.
    """
    print("\n=== Mixing final audio ===")

    duck_points = [(m["start"], m["start"] + m["target_duration"]) for m in manifest]
    vocal_filter = _build_duck_filter(duck_points)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Pseudo-stereo (haas): original on left, 20ms delayed copy on right
    # Creates natural space without changing timbre
    voice_stereo = (
        "[0]asplit[vl][vr];"
        "[vr]adelay=20[vrd];"
        "[vl][vrd]amerge,volume=2.0[en]"
    )

    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex",
        f"{voice_stereo};"
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
