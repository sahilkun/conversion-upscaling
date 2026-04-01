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

SAMPLE_RATE = 32000  # GPT-SoVITS outputs 32kHz

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dub_common


# ── Generate All Clips ─────────────────────────────────────────────────────
def generate_clips(tts, dialogues, speaker_refs):
    """Generate all TTS clips using GPT-SoVITS."""
    print("\n=== Generating English speech (GPT-SoVITS) ===")
    clips_dir = os.path.join(WORKDIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    failures_path = os.path.join(WORKDIR, "failures.txt")

    import soundfile as sf

    fallback_spk = next(iter(speaker_refs)) if speaker_refs else None
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
        word_count = len(d["text"].split())
        max_expected = max(d["duration"] * 2.5, word_count * 0.8)

        # Pre-predict English duration — English averages ~2.8 words/sec
        est_duration = word_count / 2.8
        speed_factor = 1.0
        if est_duration > d["duration"] * 1.15 and d["duration"] > 0.5:
            speed_factor = min(est_duration / d["duration"], 1.4)  # cap at 1.4x

        try:
            # Pad short text to prevent GPT-SoVITS stutter on single words
            tts_text = d["text"]
            if len(tts_text.split()) <= 2:
                tts_text = f"Well, {tts_text.lower()}"

            # Cap max tokens based on text length to prevent repetition loops
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
                "speed_factor": speed_factor,
            }

            # Generate clip — same seed always for voice consistency
            for sr, audio in tts.run(inputs):
                sf.write(clip_path, audio, sr)
                break

            # Quality check — discard if repetition loop (way too long)
            if os.path.exists(clip_path):
                with wave.open(clip_path, "rb") as _wf:
                    _clip_dur = _wf.getnframes() / _wf.getframerate()
                if _clip_dur > max_expected:
                    safe = d["text"][:40]
                    print(f"  LOOP detected line {i} ({safe}): {_clip_dur:.1f}s > {max_expected:.1f}s — skipping")
                    os.remove(clip_path)
                    failures.append(f"line {i:04d} [{d['start']:.2f}s] LOOP: {d['text'][:60]}")
                    continue

            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                manifest.append({
                    "path": clip_path, "start": d["start"],
                    "target_duration": d["duration"],
                    "emotion": emotion, "volume_factor": volume_factor,
                })
            else:
                failures.append(f"line {i:04d} [{d['start']:.2f}s] EMPTY: {d['text'][:60]}")

        except Exception as e:
            safe = d["text"][:40]
            print(f"  WARNING: line {i} ({safe}): {e}")
            failures.append(f"line {i:04d} [{d['start']:.2f}s] ERROR: {d['text'][:60]}")
            continue

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


# post_process_emotions replaced by dub_common.postprocess_clips


# assemble_voice_track and mix_final replaced by dub_common functions


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Dubbing Pipeline")
    parser.add_argument("--subs", default="subtitle.ass")
    parser.add_argument("--labels", default="speaker_labels.txt")
    parser.add_argument("--src", default=None, help="Source MKV (for Demucs)")
    parser.add_argument("--output", default=os.path.join("out", "dub_work", "english_dub.wav"))
    parser.add_argument("--workdir", default=None, help="Working directory")
    parser.add_argument("--voices-dir", default="voices", help="Default voice directory")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--llm-speakers", action="store_true", help="Use Claude API to identify character names/genders")
    args = parser.parse_args()

    global WORKDIR
    if args.workdir:
        WORKDIR = args.workdir
    else:
        WORKDIR = os.path.dirname(args.output) or os.path.join("out", "dub_work")

    os.makedirs(WORKDIR, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Parse subtitles
    print("=== Step 1: Parse subtitles ===")
    dialogues = dub_common.parse_ass(args.subs)

    # Step 2: Assign speakers (manual > LLM > ASS styles > auto-detect)
    print("\n=== Step 2: Assign speakers ===")
    labels_path = args.labels if os.path.isfile(args.labels) else None
    speaker_map = None
    if not labels_path and args.llm_speakers:
        speaker_map = dub_common.llm_label_speakers(dialogues, WORKDIR)
    dialogues = dub_common.assign_speakers(dialogues, labels_path=labels_path, speaker_map=speaker_map)

    # Step 2b: Classify emotions with full context window
    print("\n=== Step 2b: Emotion classification (context-aware) ===")
    emotion_dist = dub_common.classify_emotions_bulk(dialogues)
    print(f"  Distribution: {emotion_dist}")

    # Step 3: Demucs separation (auto-run if needed)
    vocals_path = os.path.join(WORKDIR, "vocals.wav")
    bg_path = os.path.join(WORKDIR, "background.wav")
    if not os.path.exists(vocals_path) or not os.path.exists(bg_path):
        src_mkv = args.src or dub_common.find_mkv_1080()
        if not src_mkv:
            print("ERROR: No source MKV found for Demucs. Use --src")
            sys.exit(1)
        vocals_path, bg_path = dub_common.separate_audio(src_mkv, WORKDIR)

    # Get total duration
    total_duration = dub_common.probe_duration(vocals_path)
    print(f"\n  Audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Step 4: Voice references (fallback: extracted > training_data > voices/)
    print("\n=== Step 4: Resolving voice references ===")
    voices_dir = os.path.join(base_dir, args.voices_dir) if not os.path.isabs(args.voices_dir) else args.voices_dir
    training_dir = os.path.join(base_dir, "training_data")

    # Try extracting calm refs from vocals
    extracted = dub_common.extract_calm_refs(
        dialogues, vocals_path, WORKDIR,
        min_dur=3.0, max_dur=10.0, target_seconds=10.0
    )
    # Convert extracted {speaker: wav_path} to full ref format
    extracted_full = {spk: (path, "", []) for spk, path in extracted.items()}
    speaker_refs = dub_common.resolve_voice_refs(
        extracted_full, voices_dir, training_dir=training_dir
    )

    # Step 5: Load GPT-SoVITS
    print("\n=== Loading GPT-SoVITS model ===")
    gptsovits_dir = os.path.join(base_dir, "GPT-SoVITS")
    # Resolve all paths to absolute before chdir
    abs_subs = os.path.abspath(args.subs)
    abs_output = os.path.abspath(args.output)
    abs_workdir = os.path.abspath(WORKDIR)
    abs_vocals = os.path.abspath(vocals_path)
    abs_bg = os.path.abspath(bg_path)

    original_cwd = os.getcwd()
    os.chdir(gptsovits_dir)
    sys.path.insert(0, os.path.join(gptsovits_dir, "GPT_SoVITS"))
    sys.path.insert(0, gptsovits_dir)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")

    # Search for any fine-tuned models (not just one series)
    import glob as _glob
    trained_s2 = _glob.glob(os.path.join(gptsovits_dir, "experiments", "*", "logs_s2_v2", "G_*.pth"))
    trained_s1 = _glob.glob(os.path.join(gptsovits_dir, "experiments", "*", "logs_s1_v2", "ckpt", "*.ckpt"))
    if trained_s2 and trained_s1:
        config.configs["t2s_weights_path"] = sorted(trained_s1)[-1]
        config.configs["vits_weights_path"] = sorted(trained_s2)[-1]
        print(f"  Using FINE-TUNED models!")
        print(f"    S1: {os.path.basename(config.configs['t2s_weights_path'])}")
        print(f"    S2: {os.path.basename(config.configs['vits_weights_path'])}")
    else:
        print(f"  No fine-tuned models found, using pretrained")
    config.configs["device"] = "cuda"
    config.configs["is_half"] = False
    tts = TTS(config)
    os.chdir(original_cwd)
    WORKDIR = abs_workdir
    print("  Model loaded!")

    # Step 6: Generate clips
    if args.no_cache:
        import shutil
        clips_dir = os.path.join(WORKDIR, "clips")
        if os.path.isdir(clips_dir):
            shutil.rmtree(clips_dir)

    manifest = generate_clips(tts, dialogues, speaker_refs)

    # Step 7: Post-process + time-stretch + normalize
    dub_common.postprocess_clips(manifest)
    dub_common.time_stretch_clips(manifest)
    dub_common.normalize_clips(manifest)

    # Step 7b: Resolve overlapping dialogue lines
    dub_common.resolve_overlaps(manifest)

    # Step 8: Assemble voice track
    voice_path = dub_common.assemble_voice_track(manifest, total_duration, WORKDIR)

    # Step 9: Mix final audio
    dub_common.mix_audio(voice_path, abs_vocals, abs_bg, manifest, abs_output)

    # Step 10: QA report + listening samples
    dub_common.generate_qa_report(manifest, dialogues, WORKDIR, total_duration)
    dub_common.export_listen_samples(abs_output, WORKDIR)

    print(f"\n=== DONE! Output: {abs_output} ===")


if __name__ == "__main__":
    main()
