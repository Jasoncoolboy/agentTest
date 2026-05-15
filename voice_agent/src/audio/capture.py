"""Audio capture module using sounddevice.

Provides continuous microphone input as 30ms frames (480 samples at 16kHz).
Frames are pushed to an asyncio queue for downstream consumers.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd

from src.utils.exceptions import AudioDeviceError

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from microphone and distributes frames via async queue."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 480,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    def start(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Start audio capture, pushing frames to the given queue."""
        self._queue = queue
        self._loop = loop
        self._running = True

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                f"Audio capture started: {self.sample_rate}Hz, "
                f"{self.block_size} samples/frame, device={self.device or 'default'}"
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to open audio input: {e}") from e

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Sounddevice callback - runs in PortAudio thread."""
        if status:
            logger.warning(f"Audio input status: {status}")

        if self._running and self._queue is not None and self._loop is not None:
            # Copy the data since indata buffer is reused
            frame = indata[:, 0].copy()
            self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    @staticmethod
    def list_devices() -> str:
        """List available audio input devices."""
        devices = sd.query_devices()
        lines = ["Available audio devices:"]
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[0] else ""
                lines.append(f"  [{i}] {dev['name']} (in: {dev['max_input_channels']}ch){marker}")
        return "\n".join(lines)
