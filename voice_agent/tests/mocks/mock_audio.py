import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class MockAudioCapture:
    """Mocks AudioCapture by streaming predefined audio frames or zeros."""

    def __init__(self, sample_rate=16000, block_size=480, device=None, audio_data=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_data = audio_data  # 1D numpy array
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._task = None

    def start(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop
        self._running = True
        self._task = self._loop.create_task(self._push_frames())
        logger.info("MockAudioCapture started")

    async def _push_frames(self):
        offset = 0
        while self._running:
            if self.audio_data is not None and offset < len(self.audio_data):
                end = min(offset + self.block_size, len(self.audio_data))
                frame = self.audio_data[offset:end]
                # Pad if necessary
                if len(frame) < self.block_size:
                    frame = np.pad(frame, (0, self.block_size - len(frame)))
                offset = end
            else:
                # Produce silence
                frame = np.zeros(self.block_size, dtype="float32")
                # Wait briefly to not flood the queue instantly with zeros
                await asyncio.sleep(self.block_size / self.sample_rate)

            # push to queue
            if self._queue is not None:
                self._queue.put_nowait(frame)
            
            # small sleep to mimic real time capture
            if self.audio_data is not None:
                await asyncio.sleep(self.block_size / self.sample_rate)

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("MockAudioCapture stopped")

class MockAudioPlayback:
    """Mocks AudioPlayback by storing played frames into a list."""

    def __init__(self, sample_rate=22050, device=None):
        self.sample_rate = sample_rate
        self.played_audio = []
        self._playing = False
        self._interrupted = asyncio.Event()

    async def play_audio(self, audio: np.ndarray) -> bool:
        if audio is None or len(audio) == 0:
            return True

        self._playing = True
        self._interrupted.clear()
        
        # Simulate playback time
        duration = len(audio) / self.sample_rate
        chunk_time = 0.1
        
        try:
            for _ in range(int(duration / chunk_time)):
                if self._interrupted.is_set():
                    return False
                await asyncio.sleep(chunk_time)
            
            self.played_audio.append(audio)
            return True
        finally:
            self._playing = False

    def interrupt(self):
        self._interrupted.set()
        logger.debug("MockPlayback interrupted")

    @property
    def is_playing(self) -> bool:
        return self._playing
