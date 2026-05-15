"""Pipeline orchestrator - the central async state machine.

Coordinates the full voice agent pipeline:
IDLE → (wake word) → LISTENING → (VAD) → PROCESSING → (STT+LLM+TTS) → SPEAKING → IDLE

Handles barge-in detection during SPEAKING state.
"""

import asyncio
import enum
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from src.audio.capture import AudioCapture
from src.audio.playback import AudioPlayback
from src.llm.client import LLMClient
from src.llm.history import ConversationHistory
from src.llm.tools import ToolRegistry
from src.stt.whisper import WhisperSTT
from src.tts.piper import PiperTTS
from src.utils.exceptions import LLMError, LLMTimeoutError, STTError, TTSError
from src.utils.sentence_buffer import SentenceBuffer
from src.vad.silero import SileroVAD
from src.wake_word.detector import WakeWordDetector

logger = logging.getLogger(__name__)


class AgentState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SHUTDOWN = "shutdown"


class Orchestrator:
    """Main pipeline orchestrator managing the voice agent state machine."""

    def __init__(self, config: dict):
        self.config = config
        self._state = AgentState.IDLE
        self._running = False

        # Thread pool for blocking operations (STT, TTS)
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Audio
        self._audio_capture = AudioCapture(
            sample_rate=config["audio"]["sample_rate"],
            block_size=config["audio"]["block_size"],
            device=config["audio"].get("device"),
        )
        self._audio_playback = AudioPlayback(
            sample_rate=config["audio"]["playback_sample_rate"],
            device=config["audio"].get("playback_device"),
        )

        # Wake word
        self._wake_word = WakeWordDetector(
            model_name=config["wake_word"]["model_name"],
            threshold=config["wake_word"]["threshold"],
            debounce_seconds=config["wake_word"]["debounce_seconds"],
        )

        # VAD
        self._vad = SileroVAD(
            model_path=config["vad"]["model_path"],
            speech_threshold=config["vad"]["speech_threshold"],
            silence_timeout_ms=config["vad"]["silence_timeout_ms"],
            min_speech_duration_ms=config["vad"]["min_speech_duration_ms"],
        )

        # Barge-in VAD (separate instance with different thresholds)
        self._barge_in_vad = SileroVAD(
            model_path=config["vad"]["model_path"],
            speech_threshold=config["barge_in"]["speech_threshold"],
            silence_timeout_ms=100,  # Short timeout for barge-in
            min_speech_duration_ms=config["barge_in"]["trigger_duration_ms"],
        )

        # STT (lazy loaded)
        self._stt = WhisperSTT(
            model=config["stt"]["model"],
            compute_type=config["stt"]["compute_type"],
            beam_size=config["stt"]["beam_size"],
            language=config["stt"]["language"],
            vad_filter=config["stt"]["vad_filter"],
        )

        # LLM
        self._llm = LLMClient(
            api_key_env=config["llm"]["api_key_env"],
            base_url=config["llm"]["base_url"],
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
            timeout_seconds=config["llm"]["timeout_seconds"],
            max_retries=config["llm"]["max_retries"],
        )

        # History
        self._history = ConversationHistory(
            max_tokens=config["history"]["max_tokens"],
            max_turns=config["history"]["max_turns"],
        )
        self._history.set_system_prompt(config["llm"]["system_prompt"])

        # TTS (lazy loaded)
        self._tts = PiperTTS(
            model=config["tts"]["model"],
            model_path=config["tts"]["model_path"],
            speaker_id=config["tts"].get("speaker_id"),
        )

        # Tool registry
        self._tool_registry = ToolRegistry()

        # Sentence buffer
        self._sentence_buffer = SentenceBuffer(
            min_length=config["pipeline"]["sentence_min_length"],
        )

        # Async primitives
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._barge_in_event = asyncio.Event()
        self._wake_detected_event = asyncio.Event()
        self._speech_ended_event = asyncio.Event()

        # Audio buffer for recording
        self._recording_buffer: List[np.ndarray] = []

        # Counters
        self._empty_transcription_count = 0

        # Barge-in config
        self._barge_in_mode = config["barge_in"]["mode"]

    def register_tool(self, tool):
        """Register a tool with the agent."""
        self._tool_registry.register(tool)

    async def start(self):
        """Start the voice agent pipeline."""
        self._running = True
        loop = asyncio.get_event_loop()

        # Load models that are needed immediately
        logger.info("Loading wake word model...")
        self._wake_word.load()

        logger.info("Loading VAD model...")
        self._vad.load()
        self._barge_in_vad.load()

        # Start audio capture
        logger.info("Starting audio capture...")
        self._audio_capture.start(self._audio_queue, loop)

        logger.info("Voice agent started. Listening for wake word...")
        self._set_state(AgentState.IDLE)

        # Run the main pipeline
        try:
            await self._run_pipeline()
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Stop the voice agent."""
        self._running = False
        self._set_state(AgentState.SHUTDOWN)
        self._audio_capture.stop()
        self._executor.shutdown(wait=False)
        logger.info("Voice agent stopped")

    async def _run_pipeline(self):
        """Main pipeline loop processing audio frames."""
        while self._running:
            try:
                # Get next audio frame (with timeout to allow checking _running)
                try:
                    frame = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Route frame based on current state
                if self._state == AgentState.IDLE:
                    await self._process_idle(frame)
                elif self._state == AgentState.LISTENING:
                    await self._process_listening(frame)
                elif self._state == AgentState.SPEAKING:
                    await self._process_speaking(frame)

            except Exception as e:
                logger.error(f"Pipeline loop error: {e}", exc_info=True)
                # Don't crash the loop
                await asyncio.sleep(0.1)

    async def _process_idle(self, frame: np.ndarray):
        """Process frame in IDLE state - check for wake word."""
        detected = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._wake_word.process_frame, frame
        )

        if detected:
            logger.info("*** Wake word detected! ***")
            self._set_state(AgentState.LISTENING)
            self._recording_buffer.clear()
            self._vad.reset_state()

    async def _process_listening(self, frame: np.ndarray):
        """Process frame in LISTENING state - VAD + buffer audio."""
        speech_started, speech_ended = self._vad.detect_speech_segment(frame)

        # Always buffer audio once we're listening
        if self._vad.is_speech_active or speech_started:
            self._recording_buffer.append(frame)
        elif not self._recording_buffer:
            # Haven't heard speech yet - still buffer (might be onset)
            self._recording_buffer.append(frame)
            # But limit pre-speech buffer to 1 second
            max_pre_frames = int(1.0 / 0.03)  # ~33 frames
            if len(self._recording_buffer) > max_pre_frames:
                self._recording_buffer.pop(0)

        if speech_ended:
            # Speech ended, process the recording
            if self._recording_buffer:
                audio = np.concatenate(self._recording_buffer)
                self._recording_buffer.clear()
                await self._process_utterance(audio)
            else:
                self._set_state(AgentState.IDLE)

    async def _process_speaking(self, frame: np.ndarray):
        """Process frame in SPEAKING state - check for barge-in."""
        if self._barge_in_mode == "disabled":
            return

        # Check for barge-in via VAD
        speech_started, _ = self._barge_in_vad.detect_speech_segment(frame)
        if speech_started:
            logger.info("*** Barge-in detected! ***")
            self._barge_in_event.set()
            self._audio_playback.interrupt()
            self._barge_in_vad.reset_state()
            self._set_state(AgentState.LISTENING)
            self._recording_buffer.clear()
            self._vad.reset_state()

    async def _process_utterance(self, audio: np.ndarray):
        """Full processing pipeline: STT → LLM → TTS."""
        self._set_state(AgentState.PROCESSING)

        # Ensure STT is loaded (lazy loading)
        if not self._stt.is_loaded:
            logger.info("Loading STT model (first use)...")
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._stt.load
            )

        # Transcribe
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._stt.transcribe, audio
            )
        except STTError as e:
            logger.error(f"STT failed: {e}")
            self._set_state(AgentState.IDLE)
            return

        # Handle empty transcription
        if not text or not text.strip():
            self._empty_transcription_count += 1
            if self._empty_transcription_count >= 3:
                await self._speak("I'm listening. Go ahead.")
                self._empty_transcription_count = 0
            self._set_state(AgentState.IDLE)
            return

        self._empty_transcription_count = 0
        logger.info(f"User said: '{text}'")

        # Add to history
        self._history.add_user(text)

        # Stream LLM response → TTS
        await self._stream_and_speak(text)

    async def _stream_and_speak(self, user_text: str):
        """Stream LLM response, buffer sentences, synthesize and play TTS."""
        self._set_state(AgentState.SPEAKING)
        self._barge_in_event.clear()
        self._sentence_buffer.clear()

        # Ensure TTS is loaded (lazy loading)
        if not self._tts.is_loaded:
            logger.info("Loading TTS model (first use)...")
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._tts.load
            )

        full_response = ""

        try:
            async for token in self._llm.stream_response(
                self._history, self._tool_registry
            ):
                # Check barge-in
                if self._barge_in_event.is_set():
                    logger.info("Barge-in: stopping LLM stream")
                    self._sentence_buffer.clear()
                    return

                full_response += token

                # Feed to sentence buffer
                sentences = self._sentence_buffer.add_token(token)
                for sentence in sentences:
                    if self._barge_in_event.is_set():
                        return
                    await self._speak(sentence)

            # Flush remaining buffer
            remaining = self._sentence_buffer.flush()
            if remaining and not self._barge_in_event.is_set():
                await self._speak(remaining)

        except (LLMError, LLMTimeoutError) as e:
            logger.error(f"LLM error: {e}")
            await self._speak("I'm having trouble connecting. Please try again.")

        finally:
            if not self._barge_in_event.is_set():
                self._set_state(AgentState.IDLE)
                logger.info("Response complete. Listening for wake word...")

    async def _speak(self, text: str):
        """Synthesize and play a single sentence."""
        if not text or self._barge_in_event.is_set():
            return

        try:
            # Synthesize in thread pool
            audio = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._tts.synthesize, text
            )

            if audio is not None and len(audio) > 0 and not self._barge_in_event.is_set():
                logger.debug(f"Speaking: '{text[:60]}...'")
                completed = await self._audio_playback.play_audio(audio)

                if not completed:
                    logger.debug("Playback interrupted (barge-in)")

        except TTSError as e:
            logger.error(f"TTS error: {e}")

    def _set_state(self, new_state: AgentState):
        """Transition to a new state with logging."""
        if self._state != new_state:
            logger.debug(f"State: {self._state.value} → {new_state.value}")
            self._state = new_state

    @property
    def state(self) -> AgentState:
        return self._state
