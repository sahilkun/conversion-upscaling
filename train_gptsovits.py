#!/usr/bin/env python3
"""
GPT-SoVITS Fine-tuning Script
Trains both SoVITS (S2) and GPT (S1) on character voices.

Usage:
  py -3.11 train_gptsovits.py

Requires:
  - training_annotation.list (created by annotation step)
  - training_data/female/*.wav + training_data/male/*.wav
  - GPT-SoVITS pretrained models installed
"""

import io
import json
import os
import shutil
import subprocess
import sys
import yaml

# Fix Windows Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPTSOVITS_DIR = os.path.join(BASE_DIR, "GPT-SoVITS")
EXP_NAME = "natsu_to_hako"
EXP_DIR = os.path.join(GPTSOVITS_DIR, "experiments", EXP_NAME)
ANNOTATION_FILE = os.path.join(BASE_DIR, "training_annotation.list")
PRETRAINED_DIR = os.path.join(GPTSOVITS_DIR, "GPT_SoVITS", "pretrained_models")

# Pretrained model paths
PRETRAINED_S2G = os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained", "s2G2333k.pth")
PRETRAINED_S2D = os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained", "s2D2333k.pth")
PRETRAINED_S1 = os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained",
                              "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
BERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-roberta-wwm-ext-large")
HUBERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-hubert-base")


def run_cmd(cmd, desc, cwd=None, env=None):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    # Add GPT-SoVITS paths so modules can be found
    pythonpath = os.path.join(GPTSOVITS_DIR, "GPT_SoVITS")
    pythonpath += os.pathsep + GPTSOVITS_DIR
    if "PYTHONPATH" in merged_env:
        merged_env["PYTHONPATH"] = pythonpath + os.pathsep + merged_env["PYTHONPATH"]
    else:
        merged_env["PYTHONPATH"] = pythonpath
    merged_env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=cwd or GPTSOVITS_DIR, env=merged_env,
                          capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: {desc} returned code {result.returncode}")
    return result.returncode


def stage1_prepare_data():
    """Create experiment directory and copy annotation."""
    print("\n" + "=" * 60)
    print("  STAGE 1: Prepare training data")
    print("=" * 60)

    os.makedirs(EXP_DIR, exist_ok=True)

    # Convert backslashes to forward slashes for cross-platform compat
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Make paths absolute
    fixed = []
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) == 4:
            wav_path = os.path.abspath(parts[0].replace("\\", "/"))
            fixed.append(f"{wav_path}|{parts[1]}|{parts[2]}|{parts[3]}")

    annotation_out = os.path.join(EXP_DIR, "annotation.list")
    with open(annotation_out, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed))

    print(f"  Annotation: {len(fixed)} entries")
    print(f"  Saved to: {annotation_out}")
    return annotation_out


def stage2_extract_features(annotation_path):
    """Run GPT-SoVITS data preparation (3 steps)."""
    print("\n" + "=" * 60)
    print("  STAGE 2: Extract features (BERT + HuBERT + Semantic)")
    print("=" * 60)

    env = {
        "exp_name": EXP_NAME,
        "inp_text": annotation_path,
        "inp_wav_dir": os.path.join(BASE_DIR, "training_data", "all_wavs"),
        "opt_dir": EXP_DIR,
        "bert_pretrained_dir": BERT_DIR,
        "cnhubert_base_dir": HUBERT_DIR,
        "is_half": "False",
        "pretrained_s2G": PRETRAINED_S2G,
        "s2config_path": os.path.join(GPTSOVITS_DIR, "GPT_SoVITS", "configs", "s2.json"),
        "PYTHONIOENCODING": "utf-8",
        # Parallel processing params — single process
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "version": "v2",
    }

    nltk_path = os.path.join(GPTSOVITS_DIR, "nltk_data")
    if os.path.isdir(nltk_path):
        env["NLTK_DATA"] = nltk_path

    py = sys.executable

    # Step 1: Text + BERT
    rc = run_cmd([py, "-u", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
            "Step 1/3: Extract text + BERT features", env=env)
    if rc != 0:
        print("  FATAL: Step 1 failed. Cannot continue.")
        return False

    # Step 2: HuBERT + wav32k
    rc = run_cmd([py, "-u", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
            "Step 2/3: Extract HuBERT features + resample to 32kHz", env=env)
    if rc != 0:
        print("  FATAL: Step 2 failed. Cannot continue.")
        return False

    # Step 3: Semantic tokens
    rc = run_cmd([py, "-u", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
            "Step 3/3: Extract semantic tokens", env=env)
    if rc != 0:
        print("  FATAL: Step 3 failed. Cannot continue.")
        return False

    # Merge part files (scripts output -0 suffix for single-process runs)
    import glob
    for pattern, target in [
        ("2-name2text-*.txt", "2-name2text.txt"),
        ("6-name2semantic-*.tsv", "6-name2semantic.tsv"),
    ]:
        parts = sorted(glob.glob(os.path.join(EXP_DIR, pattern)))
        target_path = os.path.join(EXP_DIR, target)
        if parts and not os.path.exists(target_path):
            with open(target_path, "w", encoding="utf-8") as out:
                for p in parts:
                    with open(p, "r", encoding="utf-8") as inp:
                        out.write(inp.read())
            print(f"  Merged {len(parts)} parts -> {target}")

    return True


def stage3_train_sovits():
    """Train SoVITS (S2) model."""
    print("\n" + "=" * 60)
    print("  STAGE 3: Train SoVITS (S2)")
    print("=" * 60)

    # Load and modify s2 config
    config_src = os.path.join(GPTSOVITS_DIR, "GPT_SoVITS", "configs", "s2.json")
    with open(config_src, "r") as f:
        config = json.load(f)

    # Modify for our training
    config["train"]["batch_size"] = 2  # 4GB VRAM
    config["train"]["epochs"] = 8
    config["train"]["fp16_run"] = True
    config["train"]["learning_rate"] = 0.0001
    config["train"]["save_every_epoch"] = 4
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["pretrained_s2G"] = PRETRAINED_S2G
    config["train"]["pretrained_s2D"] = PRETRAINED_S2D
    config["train"]["gpu_numbers"] = "0"
    config["data"]["exp_dir"] = EXP_DIR
    config["data"]["training_files"] = os.path.join(EXP_DIR, "2-name2text.txt")
    config["model"]["version"] = "v2"
    config["s2_ckpt_dir"] = os.path.join(EXP_DIR, "logs_s2_v2")

    config_out = os.path.join(EXP_DIR, "s2_train_config.json")
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    os.makedirs(os.path.join(EXP_DIR, "logs_s2_v2"), exist_ok=True)

    run_cmd([sys.executable, "-u", "GPT_SoVITS/s2_train.py", "--config", config_out],
            "Training SoVITS (S2) — ~15 min")


def stage4_train_gpt():
    """Train GPT (S1) model."""
    print("\n" + "=" * 60)
    print("  STAGE 4: Train GPT (S1)")
    print("=" * 60)

    # Load and modify s1 config
    config_src = os.path.join(GPTSOVITS_DIR, "GPT_SoVITS", "configs", "s1longer-v2.yaml")
    with open(config_src, "r") as f:
        config = yaml.safe_load(f)

    # Modify for our training
    config["train"]["batch_size"] = 2  # 4GB VRAM
    config["train"]["epochs"] = 20
    config["train"]["save_every_n_epoch"] = 5
    config["train"]["precision"] = "16-mixed"

    # Add required fields
    config["train"]["exp_name"] = EXP_NAME
    config["train"]["pretrained_s1"] = PRETRAINED_S1
    config["train"]["if_save_latest"] = True
    config["train"]["if_save_every_weights"] = True
    config["train"]["gpu_numbers"] = "0"
    config["train"]["half_weights_save_dir"] = os.path.join(EXP_DIR, "logs_s1_v2")

    # These must be top-level keys (not under "data")
    config["train_semantic_path"] = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    config["train_phoneme_path"] = os.path.join(EXP_DIR, "2-name2text.txt")

    config["output_dir"] = os.path.join(EXP_DIR, "logs_s1_v2")

    config_out = os.path.join(EXP_DIR, "s1_train_config.yaml")
    with open(config_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    os.makedirs(os.path.join(EXP_DIR, "logs_s1_v2", "ckpt"), exist_ok=True)

    run_cmd([sys.executable, "-u", "GPT_SoVITS/s1_train.py", "--config_file", config_out],
            "Training GPT (S1) — ~15 min")


def main():
    print("=" * 60)
    print("  GPT-SoVITS Fine-tuning: Natsu to Hako Characters")
    print("=" * 60)
    print(f"  Experiment: {EXP_NAME}")
    print(f"  Annotation: {ANNOTATION_FILE}")

    # Check prerequisites
    if not os.path.exists(ANNOTATION_FILE):
        print(f"ERROR: {ANNOTATION_FILE} not found!")
        sys.exit(1)
    if not os.path.exists(PRETRAINED_S2G):
        print(f"ERROR: Pretrained SoVITS not found: {PRETRAINED_S2G}")
        sys.exit(1)

    annotation = stage1_prepare_data()
    if not stage2_extract_features(annotation):
        print("\nFATAL: Feature extraction failed. Cannot train.")
        sys.exit(1)
    stage3_train_sovits()
    stage4_train_gpt()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  SoVITS model: {EXP_DIR}/logs_s2_v2/")
    print(f"  GPT model:    {EXP_DIR}/logs_s1_v2/ckpt/")
    print(f"\n  To use: update gptsovits_dub.py with trained model paths")


if __name__ == "__main__":
    main()
