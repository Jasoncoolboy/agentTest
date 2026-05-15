"""Microphone test script - verify audio device works.

Usage:
    python scripts/test_microphone.py
"""

import sys
import time

import numpy as np
import sounddevice as sd


def main():
    print("=" * 50)
    print("Microphone Test")
    print("=" * 50)

    # List devices
    print("\n--- Input Devices ---")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *DEFAULT*" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")

    print("\n--- Output Devices ---")
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            marker = " *DEFAULT*" if i == sd.default.device[1] else ""
            print(f"  [{i}] {dev['name']}{marker}")

    # Record
    duration = 3
    sample_rate = 16000
    print(f"\nRecording {duration} seconds of audio...")
    print("Speak now!")

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        print(f"\n[ERROR] Recording failed: {e}")
        print("Check that your microphone is connected and permissions are granted.")
        sys.exit(1)

    # Analyze
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    print(f"\nRecording stats:")
    print(f"  RMS level: {rms:.4f} ({20 * np.log10(rms + 1e-10):.1f} dB)")
    print(f"  Peak level: {peak:.4f} ({20 * np.log10(peak + 1e-10):.1f} dB)")

    if rms < 0.001:
        print("\n[WARNING] Very low audio level - microphone may not be working")
    else:
        print("\n[OK] Microphone is capturing audio")

    # Playback
    print(f"\nPlaying back recording...")
    try:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        print("[OK] Playback complete")
    except Exception as e:
        print(f"[ERROR] Playback failed: {e}")
        sys.exit(1)

    print("\nMicrophone test passed!")


if __name__ == "__main__":
    main()
