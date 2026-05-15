"""Piper TTS module.

Uses piper-tts for fast, offline text-to-speech synthesis.
Piper runs extremely fast on CPU (~50-150ms per sentence).
"""

import io
import json
import logging
import struct
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.exceptions import TTSError, ModelNotFoundError

logger = logging.getLogger(__name__)


class PiperTTS:
    """Text-to-speech using Piper (ONNX-based, CPU-optimized).
    
    Uses the piper-tts Python package for synthesis.
    """

    def __init__(
        self,
        model: str = "en_US-lessac-medium",
        model_path: str = "models/piper",
        speaker_id: Optional[int] = None,
    ):
        self.model_name = model
        self.model_path = Path(model_path)
        self.speaker_id = speaker_id
        self._voice = None
        self._sample_rate: int = 22050

    def load(self):
        """Load the Piper voice model."""
        model_file = self.model_path / f"{self.model_name}.onnx"
        config_file = self.model_path / f"{self.model_name}.onnx.json"

        if not model_file.exists():
            raise ModelNotFoundError(
                f"Piper model not found at {model_file}. "
                "Run: python scripts/download_models.py"
            )
        if not config_file.exists():
            raise ModelNotFoundError(
                f"Piper config not found at {config_file}. "
                "Run: python scripts/download_models.py"
            )

        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(model_file), str(config_file))

            # Read sample rate from config
            with open(config_file) as f:
                config = json.load(f)
                self._sample_rate = config.get("audio", {}).get("sample_rate", 22050)

            logger.info(
                f"Piper TTS loaded: {self.model_name} "
                f"(sample_rate={self._sample_rate})"
            )
        except ImportError:
            # Fallback: try using piper as subprocess
            logger.warning("piper Python package not available, will try subprocess fallback")
            self._voice = "subprocess"
        except Exception as e:
            raise TTSError(f"Failed to load Piper model: {e}") from e

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.
        
        Args:
            text: Text string to synthesize
            
        Returns:
            Float32 numpy array of audio samples at self.sample_rate
        """
        if self._voice is None:
            raise TTSError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        try:
            if self._voice == "subprocess":
                return self._synthesize_subprocess(text)
            else:
                return self._synthesize_python(text)
        except Exception as e:
            raise TTSError(f"Synthesis failed for '{text[:50]}...': {e}") from e

    def _synthesize_python(self, text: str) -> np.ndarray:
        """Synthesize using piper Python API."""
        # Piper outputs raw PCM via a WAV-like interface
        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, "wb") as wav_file:
            self._voice.synthesize(text, wav_file, speaker_id=self.speaker_id)

        # Read back the WAV data
        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)

        # Convert to float32 [-1.0, 1.0]
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return audio_float

    def _synthesize_subprocess(self, text: str) -> np.ndarray:
        """Fallback: synthesize using piper CLI binary."""
        model_file = self.model_path / f"{self.model_name}.onnx"

        try:
            result = subprocess.run(
                [
                    "piper",
                    "--model", str(model_file),
                    "--output-raw",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise TTSError(f"Piper subprocess failed: {result.stderr.decode()}")

            # Raw output is int16 PCM
            audio_int16 = np.frombuffer(result.stdout, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            return audio_float

        except FileNotFoundError:
            raise TTSError(
                "Piper binary not found. Install piper-tts: pip install piper-tts"
            )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None
