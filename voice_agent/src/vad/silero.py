"""Silero VAD (Voice Activity Detection) module.

Uses the Silero VAD ONNX model for lightweight, accurate
speech detection in audio frames.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.exceptions import VADError, ModelNotFoundError

logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice Activity Detection using Silero VAD ONNX model.
    
    Processes 30ms audio frames and tracks speech start/end
    with configurable thresholds and silence timeout.
    """

    def __init__(
        self,
        model_path: str = "models/silero/silero_vad.onnx",
        speech_threshold: float = 0.5,
        silence_timeout_ms: int = 700,
        min_speech_duration_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.model_path = Path(model_path)
        self.speech_threshold = speech_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = sample_rate

        self._session = None
        self._h = None
        self._c = None
        self._sr = None

        # State tracking
        self._speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._frame_duration_ms = 30  # 480 samples at 16kHz

    def load(self):
        """Load the ONNX model."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Silero VAD model not found at {self.model_path}. "
                "Run: python scripts/download_models.py"
            )

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            # Initialize hidden states
            self._reset_states()
            logger.info(f"Silero VAD loaded from {self.model_path}")
        except Exception as e:
            raise VADError(f"Failed to load Silero VAD: {e}") from e

    def _reset_states(self):
        """Reset ONNX model hidden states."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([self.sample_rate], dtype=np.int64)

    def process_frame(self, frame: np.ndarray) -> float:
        """Process a single audio frame and return speech probability.
        
        Args:
            frame: Audio frame, float32, 480 samples (30ms at 16kHz)
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self._session is None:
            raise VADError("Model not loaded. Call load() first.")

        # Ensure correct shape
        if len(frame.shape) == 1:
            frame = frame[np.newaxis, :]  # Add batch dimension

        # Run inference
        inputs = {
            "input": frame.astype(np.float32),
            "h": self._h,
            "c": self._c,
            "sr": self._sr,
        }

        output, self._h, self._c = self._session.run(None, inputs)
        probability = float(output[0][0])
        return probability

    def detect_speech_segment(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """Process frame and determine if speech started or ended.
        
        Args:
            frame: Audio frame (30ms, 16kHz, float32)
            
        Returns:
            Tuple of (speech_started, speech_ended):
            - speech_started: True on the frame where speech begins
            - speech_ended: True when silence timeout is reached after speech
        """
        probability = self.process_frame(frame)
        is_speech = probability >= self.speech_threshold

        speech_started = False
        speech_ended = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1

            # Check if speech just started (meets minimum duration)
            min_frames = self.min_speech_duration_ms // self._frame_duration_ms
            if not self._speech_active and self._speech_frames >= min_frames:
                self._speech_active = True
                speech_started = True
                logger.debug(f"Speech started (prob={probability:.2f})")
        else:
            if self._speech_active:
                self._silence_frames += 1
                # Check if silence timeout reached
                silence_ms = self._silence_frames * self._frame_duration_ms
                if silence_ms >= self.silence_timeout_ms:
                    speech_ended = True
                    self._speech_active = False
                    logger.debug(f"Speech ended (silence={silence_ms}ms)")
                    self.reset_state()
            else:
                # Not in speech, reset speech frame counter
                self._speech_frames = 0

        return speech_started, speech_ended

    def reset_state(self):
        """Reset speech tracking state (not model weights)."""
        self._speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._reset_states()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def is_speech_active(self) -> bool:
        return self._speech_active
