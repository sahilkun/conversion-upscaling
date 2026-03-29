"""Upload out/ folder to S3-compatible storage (pushr.io / r-cdn.com).

Excludes dub_work/ directory. Uploads to a named folder within a
parent series folder.

Usage:
    python upload.py "Episode 02"
    python upload.py "Episode 02" --bucket my-bucket
    python upload.py "Episode 02" --dry-run
"""

import argparse
import mimetypes
import os
import sys
import time

import boto3
from botocore.config import Config

# ── Configuration ────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
EXCLUDE_DIRS = {"dub_work"}

# Load .env file (simple parser, no dependencies)
def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())

_load_env(os.path.join(os.path.dirname(__file__), ".env"))

S3_ENDPOINT  = os.environ.get("S3_ENDPOINT", "https://s3.eu-central.r-cdn.com")
S3_ACCESS    = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET    = os.environ.get("S3_SECRET_KEY", "")


def get_content_type(filepath):
    """Guess MIME type for a file, with HLS-specific overrides."""
    ext = os.path.splitext(filepath)[1].lower()
    ct_map = {
        ".m3u8": "application/vnd.apple.mpegurl",
        ".ts":   "video/mp2t",
        ".vtt":  "text/vtt",
        ".webp": "image/webp",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp4":  "video/mp4",
        ".mkv":  "video/x-matroska",
        ".aac":  "audio/aac",
        ".wav":  "audio/wav",
        ".m4s":  "video/iso.segment",
    }
    return ct_map.get(ext) or mimetypes.guess_type(filepath)[0] or "application/octet-stream"


def collect_files(out_dir):
    """Walk out/ and collect all files, excluding EXCLUDE_DIRS."""
    files = []
    for root, dirs, filenames in os.walk(out_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in filenames:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, out_dir).replace("\\", "/")
            files.append((full_path, rel_path))
    return files


def format_size(bytes_val):
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def upload(series_folder, episode_folder, bucket, dry_run=False):
    """Upload out/ contents to s3://<bucket>/<series_folder>/<episode_folder>/"""

    if not S3_ACCESS or not S3_SECRET:
        print("ERROR: S3 credentials not found. Set S3_ACCESS_KEY and S3_SECRET_KEY")
        print("       in .env or environment variables.")
        sys.exit(1)

    files = collect_files(OUT_DIR)
    if not files:
        print(f"ERROR: No files found in {OUT_DIR}")
        sys.exit(1)

    total_size = sum(os.path.getsize(f[0]) for f in files)
    print(f"Upload to: s3://{bucket}/{series_folder}/{episode_folder}/")
    print(f"Files: {len(files)} ({format_size(total_size)})")
    print(f"Excluding: {', '.join(EXCLUDE_DIRS)}")
    print()

    if dry_run:
        for full_path, rel_path in files:
            size = format_size(os.path.getsize(full_path))
            ct = get_content_type(full_path)
            key = f"{series_folder}/{episode_folder}/{rel_path}"
            print(f"  [DRY] {key}  ({size}, {ct})")
        print(f"\nDry run complete. {len(files)} files would be uploaded.")
        return

    # Create S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS,
        aws_secret_access_key=S3_SECRET,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )

    uploaded = 0
    failed = 0
    bytes_uploaded = 0
    start_time = time.time()

    for i, (full_path, rel_path) in enumerate(files, 1):
        key = f"{series_folder}/{episode_folder}/{rel_path}"
        ct = get_content_type(full_path)
        fsize = os.path.getsize(full_path)

        pct = i / len(files) * 100
        print(f"  [{i}/{len(files)} {pct:5.1f}%] {rel_path} ({format_size(fsize)}) ... ", end="", flush=True)

        try:
            extra = {"ContentType": ct}
            # Cache control: segments are immutable, playlists are not
            if rel_path.endswith((".ts", ".m4s", ".jpg", ".jpeg", ".webp")):
                extra["CacheControl"] = "public, max-age=31536000, immutable"
            elif rel_path.endswith(".m3u8"):
                extra["CacheControl"] = "public, max-age=5"
            elif rel_path.endswith(".vtt"):
                extra["CacheControl"] = "public, max-age=86400"

            s3.upload_file(full_path, bucket, key, ExtraArgs=extra)
            print("OK")
            uploaded += 1
            bytes_uploaded += fsize

        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

    elapsed = time.time() - start_time
    speed = bytes_uploaded / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"  Uploaded: {uploaded}/{len(files)} files ({format_size(bytes_uploaded)})")
    if failed:
        print(f"  Failed:   {failed}")
    print(f"  Time:     {elapsed:.1f}s ({format_size(speed)}/s)")
    print(f"  Location: s3://{bucket}/{series_folder}/{episode_folder}/")


def main():
    parser = argparse.ArgumentParser(
        description="Upload HLS output to S3-compatible storage.",
        epilog="Examples:\n"
               "  python upload.py \"Episode 02\"\n"
               "  python upload.py \"Episode 02\" --series \"My Series\"\n"
               "  python upload.py \"Episode 02\" --dry-run\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", help="Episode/batch folder name (e.g. 'Episode 02')")
    parser.add_argument("--series", default="Nagachichi Nagai-san The Animation",
                        help="Parent series folder (default: from current project)")
    parser.add_argument("--bucket", default="smallupload",
                        help="S3 bucket name (default: smallupload)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    args = parser.parse_args()

    upload(args.series, args.folder, args.bucket, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
