"""Download all required model files for the voice agent.

Run this script once before starting the agent:
    python scripts/download_models.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"


def download_file(url: str, dest: Path, description: str):
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  [SKIP] {description} already exists at {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWNLOADING] {description}...")
    print(f"    URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [DONE] {size_mb:.1f} MB downloaded")
    except Exception as e:
        print(f"  [ERROR] Failed to download {description}: {e}")
        if dest.exists():
            dest.unlink()
        raise


def download_silero_vad():
    """Download Silero VAD ONNX model."""
    print("\n--- Silero VAD ---")
    dest = MODELS_DIR / "silero" / "silero_vad.onnx"
    url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    download_file(url, dest, "Silero VAD ONNX model")


def download_openwakeword():
    """Download OpenWakeWord model (hey_jarvis)."""
    print("\n--- OpenWakeWord ---")
    # OpenWakeWord downloads models automatically on first use,
    # but we can pre-download for faster startup
    print("  [INFO] OpenWakeWord models are downloaded automatically on first use.")
    print("  [INFO] The 'hey_jarvis' model will be fetched when the agent starts.")
    print("  [INFO] Alternatively, run: python -c \"import openwakeword; openwakeword.utils.download_models()\"")


def download_whisper():
    """Download faster-whisper distil-small.en model."""
    print("\n--- Faster-Whisper (distil-small.en) ---")
    dest_dir = MODELS_DIR / "whisper"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # faster-whisper downloads from HuggingFace on first use
    # We trigger the download by importing and loading
    print("  [INFO] The faster-whisper model will be downloaded on first use.")
    print("  [INFO] Model: Systran/faster-distil-whisper-small.en")
    print("  [INFO] Size: ~150 MB (int8 quantized)")
    print("  [INFO] To pre-download, run:")
    print("    python -c \"from faster_whisper import WhisperModel; WhisperModel('distil-small.en', device='cpu', compute_type='int8')\"")


def download_piper():
    """Download Piper TTS voice model."""
    print("\n--- Piper TTS ---")
    dest_dir = MODELS_DIR / "piper"
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

    download_file(model_url, dest_dir / "en_US-lessac-medium.onnx", "Piper voice model")
    download_file(config_url, dest_dir / "en_US-lessac-medium.onnx.json", "Piper voice config")


def verify_downloads():
    """Verify all required files exist."""
    print("\n--- Verification ---")
    required = [
        MODELS_DIR / "silero" / "silero_vad.onnx",
        MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
        MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json",
    ]

    all_ok = True
    for path in required:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {path.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {path.relative_to(BASE_DIR)}")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("Voice Agent - Model Downloader")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")

    try:
        download_silero_vad()
        download_openwakeword()
        download_whisper()
        download_piper()
    except Exception as e:
        print(f"\n[FATAL] Download failed: {e}")
        sys.exit(1)

    if verify_downloads():
        print("\n[SUCCESS] All required models are ready.")
        print("\nNote: OpenWakeWord and faster-whisper models will download")
        print("automatically on first agent startup (~200 MB additional).")
    else:
        print("\n[WARNING] Some models are missing. Re-run this script.")
        sys.exit(1)

    total_size = sum(
        f.stat().st_size
        for f in MODELS_DIR.rglob("*")
        if f.is_file()
    )
    print(f"\nTotal models disk usage: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
