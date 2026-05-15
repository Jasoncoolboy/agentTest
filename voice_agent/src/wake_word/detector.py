"""OpenWakeWord wake word detector.

Listens for a configurable wake word (default: "hey_jarvis")
using the OpenWakeWord library with ONNX runtime.
"""

import logging
import time
from typing import Optional

import numpy as np

from src.utils.exceptions import WakeWordError

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects wake words in audio frames using OpenWakeWord."""

    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        debounce_seconds: float = 1.5,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.debounce_seconds = debounce_seconds
        self._model = None
        self._last_detection_time: float = 0

    def load(self):
        """Load the wake word model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            # Download models if not present
            openwakeword.utils.download_models()

            self._model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            logger.info(f"Wake word model loaded: '{self.model_name}' (threshold={self.threshold})")
        except Exception as e:
            raise WakeWordError(f"Failed to load wake word model '{self.model_name}': {e}") from e

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a single audio frame and check for wake word.
        
        Args:
            frame: Audio frame (float32, 16kHz, mono). Expected 480 samples (30ms)
                   but OpenWakeWord internally buffers to its required chunk size.
                   
        Returns:
            True if wake word detected (respecting debounce), False otherwise.
        """
        if self._model is None:
            raise WakeWordError("Model not loaded. Call load() first.")

        # OpenWakeWord expects int16 input
        audio_int16 = (frame * 32767).astype(np.int16)

        # Run prediction
        prediction = self._model.predict(audio_int16)

        # Check score against threshold
        score = prediction.get(self.model_name, 0)

        if score >= self.threshold:
            now = time.time()
            # Debounce: ignore if too soon after last detection
            if now - self._last_detection_time >= self.debounce_seconds:
                self._last_detection_time = now
                logger.info(f"Wake word detected! (score={score:.3f})")
                self._model.reset()
                return True

        return False

    def reset(self):
        """Reset the model's internal state."""
        if self._model is not None:
            self._model.reset()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
