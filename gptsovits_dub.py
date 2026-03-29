#!/usr/bin/env python3
"""
GPT-SoVITS Dubbing Pipeline — Zero-shot with XTTS pipeline improvements.

Features:
  - Manual speaker labels (from speaker_labels.txt)
  - Calmness-scored reference selection per speaker
  - Consistent voice per character (same ref for all lines)
  - Emotion-aware post-processing (volume/pitch/speed)
  - Smart vocal ducking (keeps moaning/breathing between dialogue)
  - Smooth duck transitions

Usage:
  py -3.11 gptsovits_dub.py
  py -3.11 gptsovits_dub.py --subs "subtitle hentaiclick.ass" --labels speaker_labels.txt
"""

import argparse
import array
import os
import re
import struct
import subprocess
import sys
import wave

# Add GPT-SoVITS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GPT-SoVITS", "GPT_SoVITS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GPT-SoVITS"))
os.environ["is_half"] = "False"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Fix Windows cp1252 crash when GPT-SoVITS prints Chinese characters
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
nltk_path = os.path.join(os.path.dirname(__file__), "GPT-SoVITS", "nltk_data")
if os.path.isdir(nltk_path):
    os.environ["NLTK_DATA"] = nltk_path

WORKDIR = "out/dub_work"
SAMPLE_RATE = 32000  # GPT-SoVITS outputs 32kHz


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
    return dialogues


# ── Speaker Labels Parser ──────────────────────────────────────────────────
def load_speaker_labels(labels_path, dialogues):
    """Load manual speaker labels and assign to dialogues."""
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

    # Assign labels to dialogues
    for i, d in enumerate(dialogues):
        d["speaker"] = labels.get(i + 2, "female")  # +2 offset for disclaimer lines

    f_count = sum(1 for d in dialogues if d["speaker"] == "female")
    m_count = sum(1 for d in dialogues if d["speaker"] == "male")
    print(f"  Female: {f_count} lines, Male: {m_count} lines")
    return dialogues


# ── Calmness Measurement ───────────────────────────────────────────────────
def measure_calmness(wav_path):
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


# ── Extract Calm References ────────────────────────────────────────────────
def extract_calm_refs(dialogues, vocals_path):
    """Extract calmest 3-10s vocal clips per speaker for GPT-SoVITS.

    Returns primary ref + auxiliary refs for multi-speaker tone fusion.
    """
    print("\n=== Extracting calm voice references ===")
    refs_dir = os.path.join(WORKDIR, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    MIN_DUR = 3.0
    MAX_DUR = 10.0  # GPT-SoVITS requires 3-10s
    MAX_CANDIDATES = 10

    speaker_candidates = {}
    for i, d in enumerate(dialogues):
        if d["duration"] < MIN_DUR or d["duration"] > MAX_DUR:
            continue
        spk = d["speaker"]
        if spk not in speaker_candidates:
            speaker_candidates[spk] = []
        if len(speaker_candidates[spk]) < MAX_CANDIDATES:
            speaker_candidates[spk].append((i, d))

    speaker_refs = {}  # speaker -> (wav_path, ref_text, aux_paths)
    for spk, candidates in speaker_candidates.items():
        print(f"  {spk}: evaluating {len(candidates)} candidates...")

        # Score all candidates
        scored = []
        for idx, (i, d) in enumerate(candidates):
            ref_path = os.path.join(refs_dir, f"{spk}_cand{idx}.wav")
            pad = 0.1
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
                scored.append((score, ref_path, d["text"], idx))

        if not scored:
            continue

        # Sort by calmness (lower = calmer)
        scored.sort(key=lambda x: x[0])

        # Primary ref = calmest clip
        import shutil
        best = scored[0]
        primary_ref = os.path.join(refs_dir, f"{spk}.wav")
        shutil.copy2(best[1], primary_ref)

        # Auxiliary refs = next 2-3 calmest clips (for tone fusion)
        aux_paths = []
        for j in range(1, min(4, len(scored))):
            aux_path = os.path.join(refs_dir, f"{spk}_aux{j}.wav")
            shutil.copy2(scored[j][1], aux_path)
            aux_paths.append(aux_path)

        speaker_refs[spk] = (primary_ref, best[2], aux_paths)
        print(f"    -> {spk}: primary calmness={best[0]:.1f}, +{len(aux_paths)} aux refs")
        print(f"       text=\"{best[2][:50]}\"")

    # Clean up candidate files
    for f in os.listdir(refs_dir):
        if "_cand" in f:
            os.remove(os.path.join(refs_dir, f))

    return speaker_refs


# ── Emotion Classification ─────────────────────────────────────────────────
def classify_emotion(text):
    tl = text.lower().rstrip()
    if re.search(r"cumming|cum!|shooting|there!|it.s coming", tl) or (tl.endswith("!") and len(tl) < 25):
        return "exclaim", 1.1, 1.3
    if re.search(r"^(yes|okay|huh|please)\b", tl) and len(tl) < 20:
        return "whisper", 0.9, 0.75
    if tl.endswith("?"):
        return "question", 1.0, 1.0
    if re.search(r"can.t stop|so good|feels|deeper|harder|faster|don.t pull", tl):
        return "intense", 1.05, 1.15
    return "neutral", 1.0, 1.0


# ── Generate All Clips ─────────────────────────────────────────────────────
def generate_clips(tts, dialogues, speaker_refs):
    """Generate all TTS clips using GPT-SoVITS."""
    print("\n=== Generating English speech (GPT-SoVITS) ===")
    clips_dir = os.path.join(WORKDIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    import soundfile as sf

    fallback_spk = next(iter(speaker_refs)) if speaker_refs else None
    manifest = []
    emotions = {}

    for i, d in enumerate(dialogues):
        clip_path = os.path.join(clips_dir, f"line_{i:04d}.wav")
        emotion, speed_factor, volume_factor = classify_emotion(d["text"])
        emotions[emotion] = emotions.get(emotion, 0) + 1

        # Skip if already generated (resume support)
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            manifest.append({
                "path": clip_path, "start": d["start"],
                "target_duration": d["duration"],
                "emotion": emotion, "volume_factor": volume_factor,
            })
            continue

        if d["duration"] < 0.3:
            continue

        spk = d["speaker"]
        if spk not in speaker_refs:
            spk = fallback_spk
        if not spk:
            continue

        ref_path, ref_text, aux_paths = speaker_refs[spk]

        try:
            # Pad short text to prevent GPT-SoVITS stutter on single words
            tts_text = d["text"]
            if len(tts_text.split()) <= 2:
                tts_text = f"Well, {tts_text.lower()}"

            # Cap max tokens based on text length to prevent repetition loops
            # ~10 tokens per word is generous; short texts get fewer tokens
            word_count = len(tts_text.split())
            max_tokens = min(max(word_count * 15, 50), 1024)

            inputs = {
                "text": tts_text,
                "text_lang": "en",
                "ref_audio_path": ref_path,
                "aux_ref_audio_paths": aux_paths,
                "prompt_text": ref_text,
                "prompt_lang": "en",
                "top_k": 15,
                "top_p": 0.8,
                "temperature": 0.6,
                "seed": 42,
                "repetition_penalty": 2.0,
                "max_new_tokens": max_tokens,
                "speed_factor": 1.0,
            }

            # Try up to 3 times with different seeds if output is weak
            for attempt in range(3):
                if attempt > 0:
                    inputs["seed"] = 42 + attempt * 100
                for sr, audio in tts.run(inputs):
                    sf.write(clip_path, audio, sr)
                    break
                # Quality check — peak + duration sanity
                if os.path.exists(clip_path):
                    import struct as _struct
                    with wave.open(clip_path, "rb") as _wf:
                        _nframes = _wf.getnframes()
                        _sr = _wf.getframerate()
                        _raw = _wf.readframes(_nframes)
                    _clip_dur = _nframes / _sr
                    _samples = _struct.unpack(f"<{len(_raw)//2}h", _raw)
                    _peak = max(abs(s) for s in _samples) if _samples else 0
                    # Fail if: too quiet, or way too long (repetition loop)
                    _max_expected = max(d["duration"] * 2.5, len(tts_text.split()) * 0.8)
                    if _peak >= 6000 and _clip_dur <= _max_expected:
                        break  # Good clip
                    elif attempt < 2:
                        os.remove(clip_path)
                        continue  # Retry

            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                manifest.append({
                    "path": clip_path, "start": d["start"],
                    "target_duration": d["duration"],
                    "emotion": emotion, "volume_factor": volume_factor,
                })

        except Exception as e:
            safe = d["text"][:40]
            print(f"  WARNING: line {i} ({safe}): {e}")
            continue

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(dialogues)} | emotions: {emotions}")

    print(f"  Generated {len(manifest)} clips | emotions: {emotions}")
    return manifest


# ── Emotion Post-Processing ────────────────────────────────────────────────
def post_process_emotions(manifest):
    """Apply volume/pitch adjustments based on emotion."""
    print("\n=== Post-processing emotions ===")
    processed = 0
    for m in manifest:
        if m["emotion"] == "neutral":
            continue

        raw = m["path"]
        tmp = raw + ".tmp.wav"

        af_filters = []
        if m["emotion"] == "exclaim":
            af_filters.append("volume=1.3")
        elif m["emotion"] == "whisper":
            af_filters.append("volume=0.75")
            af_filters.append("lowpass=f=6000")
        elif m["emotion"] == "intense":
            af_filters.append("volume=1.15")

        if af_filters:
            subprocess.run([
                "ffmpeg", "-y", "-i", raw,
                "-af", ",".join(af_filters), tmp
            ], capture_output=True)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                os.replace(tmp, raw)
                processed += 1
            elif os.path.exists(tmp):
                os.remove(tmp)

    print(f"  Post-processed {processed} clips")


# ── Assemble Voice Track ───────────────────────────────────────────────────
def assemble_voice_track(manifest, total_duration):
    """Place all clips at their correct timestamps on a silent timeline."""
    print("\n=== Assembling English voice track ===")

    num_samples = int(total_duration * SAMPLE_RATE)
    timeline = array.array("d", [0.0]) * num_samples

    for m in manifest:
        try:
            with wave.open(m["path"], "rb") as wf:
                clip_sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                raw = wf.readframes(n_frames)

            if wf.getsampwidth() == 2:
                samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
            else:
                continue

            # Convert to mono if stereo
            if n_channels == 2:
                mono = [(samples[j] + samples[j + 1]) / 2 for j in range(0, len(samples), 2)]
            else:
                mono = list(samples)

            # Resample if needed
            if clip_sr != SAMPLE_RATE:
                ratio = SAMPLE_RATE / clip_sr
                new_len = int(len(mono) * ratio)
                resampled = []
                for j in range(new_len):
                    src_idx = j / ratio
                    idx = int(src_idx)
                    if idx < len(mono) - 1:
                        frac = src_idx - idx
                        resampled.append(mono[idx] * (1 - frac) + mono[idx + 1] * frac)
                    elif idx < len(mono):
                        resampled.append(mono[idx])
                mono = resampled

            # No in-memory speedup — handled by FFmpeg atempo (pitch-preserving)
            pass

            # Place on timeline
            start_sample = int(m["start"] * SAMPLE_RATE)
            for j, s in enumerate(mono):
                pos = start_sample + j
                if 0 <= pos < num_samples:
                    timeline[pos] += s

        except Exception as e:
            continue

    # Normalize
    peak = max(abs(s) for s in timeline) if timeline else 1
    if peak > 0:
        scale = 30000 / peak
        for j in range(len(timeline)):
            timeline[j] = int(timeline[j] * scale)

    # Write WAV
    voice_path = os.path.join(WORKDIR, "english_voice.wav")
    with wave.open(voice_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(timeline)}h", *[int(s) for s in timeline]))

    print(f"  Voice track: {voice_path} ({total_duration:.1f}s)")
    return voice_path


# ── Smart Audio Mixing ─────────────────────────────────────────────────────
def mix_final(voice_path, vocals_path, bg_path, manifest, total_duration, output_path):
    """Mix English voice + ducked Japanese vocals + background."""
    print("\n=== Mixing final audio ===")

    # Build ducking filter
    duck_points = []
    for m in manifest:
        duck_points.append((m["start"], m["start"] + m["target_duration"]))

    if duck_points:
        parts = []
        for s, e in duck_points:
            fade = 0.2
            parts.append(f"between(t,{s-fade:.2f},{e+fade:.2f})")
        duck_expr = "+".join(parts)
        # Mute Japanese vocals completely during English speech (0.0)
        # Keep them at 0.6 between dialogue (moaning/breathing)
        vocal_filter = f"volume='if({duck_expr},0.0,0.6)':eval=frame"
    else:
        vocal_filter = "volume=0.6"

    # Resample voice to 44100 for mixing
    voice_44k = voice_path + ".44k.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path,
        "-ar", "44100", "-ac", "1", voice_44k
    ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_44k,
        "-i", vocals_path,
        "-i", bg_path,
        "-filter_complex",
        f"[0]volume=1.5[en];"
        f"[1]{vocal_filter}[ja];"
        f"[2]volume=0.8[bg];"
        f"[en][ja][bg]amix=inputs=3:duration=longest:normalize=0[out]",
        "-map", "[out]",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final dub: {output_path} ({size_mb:.1f} MB)")

    # Cleanup
    if os.path.exists(voice_44k):
        os.remove(voice_44k)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Dubbing Pipeline")
    parser.add_argument("--subs", default="subtitle hentaiclick.ass")
    parser.add_argument("--labels", default="speaker_labels.txt")
    parser.add_argument("--src", default=None, help="Source MKV (for Demucs)")
    parser.add_argument("--output", default="out/dub_work/english_dub.wav")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(WORKDIR, exist_ok=True)

    # Step 1: Parse subtitles
    print("=== Step 1: Parse subtitles ===")
    dialogues = parse_ass(args.subs)
    print(f"  {len(dialogues)} dialogue lines")

    # Step 2: Load speaker labels
    print("\n=== Step 2: Load speaker labels ===")
    dialogues = load_speaker_labels(args.labels, dialogues)

    # Step 3: Check for Demucs-separated audio
    vocals_path = os.path.join(WORKDIR, "vocals.wav")
    bg_path = os.path.join(WORKDIR, "background.wav")
    if not os.path.exists(vocals_path) or not os.path.exists(bg_path):
        print("\nERROR: Run Demucs separation first!")
        print(f"  Missing: {vocals_path} and/or {bg_path}")
        sys.exit(1)

    # Get total duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", vocals_path],
        capture_output=True, text=True
    )
    total_duration = float(result.stdout.strip())
    print(f"\n  Audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Step 4: Use known-clean reference clips (tested and confirmed)
    print("\n=== Step 4: Loading clean voice references ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    speaker_refs = {
        "female": (
            os.path.join(base_dir, "training_data", "female", "line_018.wav"),
            "Well, my parents ran away to get married, so we dont know each other.",
            [],
        ),
        "male": (
            os.path.join(base_dir, "training_data", "male", "line_044.wav"),
            "Seeing a brat like you naked does not excite me at all.",
            [],
        ),
    }
    for spk, (ref, txt, _) in speaker_refs.items():
        print(f"  {spk}: {os.path.basename(ref)} - \"{txt[:50]}\"")


    # Step 5: Load GPT-SoVITS (must cd into its directory for module imports)
    print("\n=== Loading GPT-SoVITS model ===")
    gptsovits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT-SoVITS")
    original_cwd = os.getcwd()
    os.chdir(gptsovits_dir)
    sys.path.insert(0, os.path.join(gptsovits_dir, "GPT_SoVITS"))
    sys.path.insert(0, gptsovits_dir)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    # Use fine-tuned models from training
    trained_s2 = os.path.join(gptsovits_dir, "experiments", "natsu_to_hako", "logs_s2_v2", "G_233333333333.pth")
    trained_s1 = os.path.join(gptsovits_dir, "experiments", "natsu_to_hako", "logs_s1_v2", "ckpt", "epoch=19-step=500.ckpt")
    if os.path.exists(trained_s2) and os.path.exists(trained_s1):
        config.configs["t2s_weights_path"] = trained_s1
        config.configs["vits_weights_path"] = trained_s2
        print(f"  Using FINE-TUNED models!")
    else:
        print(f"  WARNING: Trained models not found, using pretrained")
    config.configs["device"] = "cuda"
    config.configs["is_half"] = False
    tts = TTS(config)
    os.chdir(original_cwd)
    print("  Model loaded!")

    # Step 6: Generate clips
    if args.no_cache:
        import shutil
        clips_dir = os.path.join(WORKDIR, "clips")
        if os.path.isdir(clips_dir):
            shutil.rmtree(clips_dir)

    manifest = generate_clips(tts, dialogues, speaker_refs)

    # Step 7: Post-process emotions
    post_process_emotions(manifest)

    # Step 7b: Pitch-preserving time-stretch for clips that are too long
    print("\n=== Time-stretching long clips (pitch-preserving) ===")
    stretched = 0
    for m in manifest:
        clip = m["path"]
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", clip],
            capture_output=True, text=True
        )
        try:
            clip_dur = float(result.stdout.strip())
        except:
            continue

        target = m["target_duration"]
        if clip_dur > target * 1.2 and target > 0.3:
            ratio = clip_dur / target
            # Cap at 1.15x speedup — preserve natural pacing, allow overlap
            ratio = min(ratio, 1.15)
            tmp = clip + ".tempo.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", clip,
                "-af", f"atempo={ratio:.3f}",
                tmp
            ], capture_output=True)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                os.replace(tmp, clip)
                stretched += 1
            elif os.path.exists(tmp):
                os.remove(tmp)
    print(f"  Stretched {stretched} clips (max 1.15x, pitch preserved)")

    # Step 8: Assemble voice track
    voice_path = assemble_voice_track(manifest, total_duration)

    # Step 9: Mix final audio
    mix_final(voice_path, vocals_path, bg_path, manifest, total_duration, args.output)

    print(f"\n=== DONE! Output: {args.output} ===")


if __name__ == "__main__":
    main()
