"""
download_model.py
=================
Downloads the GGUF model required for offline robot safety checking.

Uses huggingface_hub to pull a quantised Qwen2.5-3B-Instruct (Q4_K_M) model
into the local `models/` directory.  Run this ONCE with internet access:

    python download_model.py

After that, safety_checker.py works entirely offline.
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# ─── Configuration ────────────────────────────────────────────────────────────
REPO_ID   = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME  = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_DIR = Path(__file__).parent / "models"


def main() -> None:
    LOCAL_DIR.mkdir(exist_ok=True)
    dest = LOCAL_DIR / FILENAME

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {dest}  ({size_mb:.0f} MB)")
        print("Delete the file and re-run this script if you want a fresh download.")
        return

    print(f"Downloading {REPO_ID} / {FILENAME}")
    print(f"Destination: {LOCAL_DIR.resolve()}")
    print("This is a one-time ~2 GB download.  Please wait...\n")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
    )

    size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
    print(f"\nDone!  Model saved to: {dest}  ({size_mb:.0f} MB)")
    print("You can now run:  python safety_checker.py")


if __name__ == "__main__":
    main()
