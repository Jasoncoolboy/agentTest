"""Speech-to-Text module using faster-whisper.

Wraps faster-whisper with distil-small.en model for CPU-efficient
transcription with int8 quantization.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.exceptions import STTError

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Speech-to-text using faster-whisper (CTranslate2 backend)."""

    def __init__(
        self,
        model: str = "distil-small.en",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
        vad_filter: bool = True,
        max_audio_duration_s: int = 30,
    ):
        self.model_name = model
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter
        self.max_audio_duration_s = max_audio_duration_s
        self._model = None

    def load(self):
        """Load the whisper model. This may take a few seconds and ~150MB RAM."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading whisper model '{self.model_name}' (compute_type={self.compute_type})...")
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            raise STTError(f"Failed to load whisper model: {e}") from e

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.
        
        Args:
            audio: Float32 numpy array of audio samples
            sample_rate: Sample rate of the audio (should be 16000)
            
        Returns:
            Transcribed text string, or empty string if nothing detected.
        """
        if self._model is None:
            raise STTError("Model not loaded. Call load() first.")

        if len(audio) == 0:
            return ""

        # Trim to max duration
        max_samples = self.max_audio_duration_s * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.warning(f"Audio trimmed to {self.max_audio_duration_s}s")

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=self.vad_filter,
                without_timestamps=True,
            )

            # Concatenate all segments
            text = " ".join(segment.text.strip() for segment in segments)
            text = text.strip()

            if text:
                logger.info(f"Transcription ({info.duration:.1f}s audio): '{text}'")
            else:
                logger.debug("Empty transcription (silence or noise)")

            return text

        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
