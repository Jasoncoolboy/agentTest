"""Audio playback module using sounddevice.

Plays synthesized audio chunks from a queue with support for
immediate interruption (barge-in).
"""

import asyncio
import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from src.utils.exceptions import AudioDeviceError

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Plays audio chunks from an async queue with interrupt support."""

    def __init__(
        self,
        sample_rate: int = 22050,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self._playing = False
        self._interrupted = threading.Event()

    async def play_audio(self, audio: np.ndarray) -> bool:
        """Play an audio array. Returns False if interrupted.
        
        Args:
            audio: numpy array of float32 audio samples
            
        Returns:
            True if playback completed, False if interrupted
        """
        if audio is None or len(audio) == 0:
            return True

        self._playing = True
        self._interrupted.clear()

        try:
            # Run blocking playback in thread executor
            loop = asyncio.get_event_loop()
            completed = await loop.run_in_executor(None, self._play_blocking, audio)
            return completed
        except Exception as e:
            logger.error(f"Playback error: {e}")
            return False
        finally:
            self._playing = False

    def _play_blocking(self, audio: np.ndarray) -> bool:
        """Blocking playback with interrupt checking."""
        # Play in chunks to allow interrupt checking
        chunk_size = self.sample_rate // 10  # 100ms chunks
        offset = 0

        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            stream.start()

            while offset < len(audio):
                if self._interrupted.is_set():
                    stream.stop()
                    stream.close()
                    return False

                end = min(offset + chunk_size, len(audio))
                chunk = audio[offset:end]
                stream.write(chunk.reshape(-1, 1))
                offset = end

            stream.stop()
            stream.close()
            return True

        except Exception as e:
            raise AudioDeviceError(f"Playback failed: {e}") from e

    def interrupt(self):
        """Immediately stop playback (called during barge-in)."""
        self._interrupted.set()
        logger.debug("Playback interrupted")

    @property
    def is_playing(self) -> bool:
        return self._playing

    @staticmethod
    def list_devices() -> str:
        """List available audio output devices."""
        devices = sd.query_devices()
        lines = ["Available audio output devices:"]
        for i, dev in enumerate(devices):
            if dev["max_output_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[1] else ""
                lines.append(f"  [{i}] {dev['name']} (out: {dev['max_output_channels']}ch){marker}")
        return "\n".join(lines)
