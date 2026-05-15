import asyncio
import numpy as np
import pytest

from src.orchestrator import Orchestrator, AgentState
from tests.mocks.mock_audio import MockAudioCapture, MockAudioPlayback
from tests.mocks.mock_llm import MockLLMClient

@pytest.fixture
def mock_pipeline_dependencies(mocker, mock_config):
    # Patch AudioCapture and Playback with our mocks
    mock_capture_class = mocker.patch("src.orchestrator.AudioCapture")
    mock_capture = MockAudioCapture(audio_data=None) # Start with silence
    mock_capture_class.return_value = mock_capture

    mock_playback_class = mocker.patch("src.orchestrator.AudioPlayback")
    mock_playback = MockAudioPlayback()
    mock_playback_class.return_value = mock_playback

    # Patch LLM Client
    mock_llm_class = mocker.patch("src.orchestrator.LLMClient")
    mock_llm = MockLLMClient(predefined_responses=["Mock response from LLM."])
    mock_llm_class.return_value = mock_llm

    # Patch STT, TTS, VAD, WakeWord for fast execution without model weights
    mock_stt = mocker.patch("src.orchestrator.WhisperSTT")
    mock_stt.return_value.is_loaded = True
    mock_stt.return_value.transcribe.return_value = "hello jarvis"
    
    mock_tts = mocker.patch("src.orchestrator.PiperTTS")
    mock_tts.return_value.is_loaded = True
    mock_tts.return_value.synthesize.return_value = np.zeros(1000, dtype="float32") # Mock TTS audio
    
    mock_vad = mocker.patch("src.orchestrator.SileroVAD")
    mock_vad.return_value.is_speech_active = False
    # Mock VAD to not detect speech by default
    mock_vad.return_value.detect_speech_segment.return_value = (False, False)

    mock_barge_in_vad = mocker.patch("src.orchestrator.SileroVAD")
    mock_barge_in_vad.return_value.is_speech_active = False
    mock_barge_in_vad.return_value.detect_speech_segment.return_value = (False, False)

    # Make SileroVAD instances distinct if needed
    mocker.patch("src.orchestrator.SileroVAD", side_effect=[mock_vad.return_value, mock_barge_in_vad.return_value])

    mock_wakeword = mocker.patch("src.orchestrator.WakeWordDetector")
    mock_wakeword.return_value.process_frame.return_value = False

    return {
        "capture": mock_capture,
        "playback": mock_playback,
        "llm": mock_llm,
        "stt": mock_stt.return_value,
        "tts": mock_tts.return_value,
        "vad": mock_vad.return_value,
        "wakeword": mock_wakeword.return_value
    }

@pytest.mark.asyncio
async def test_e2e_pipeline_execution(mock_config, mock_pipeline_dependencies):
    """
    Test a full turn of the agent: 
    wake word -> listening -> VAD detects speech end -> processing -> speaking -> idle
    """
    orchestrator = Orchestrator(mock_config)
    
    # We will simulate the events by directly calling internal methods
    # to avoid the complexities of `await orchestrator.start()` running forever.
    # Start capture manually since we bypassed `start()`
    loop = asyncio.get_event_loop()
    mock_pipeline_dependencies["capture"].start(orchestrator._audio_queue, loop)
    orchestrator._running = True

    # Initial state
    assert orchestrator.state == AgentState.IDLE

    # 1. Trigger Wake Word
    mock_pipeline_dependencies["wakeword"].process_frame.return_value = True
    frame = np.zeros(480, dtype="float32")
    await orchestrator._process_idle(frame)
    assert orchestrator.state == AgentState.LISTENING

    # 2. Trigger Listening / VAD Speech Ended
    # Simulate speech ending (started=False, ended=True)
    mock_pipeline_dependencies["vad"].detect_speech_segment.return_value = (False, True)
    # Give it some audio in the buffer
    orchestrator._recording_buffer.append(np.zeros(480, dtype="float32"))
    
    # Process the frame in listening state, which should trigger utterance processing
    # _process_listening calls _process_utterance which calls _stream_and_speak
    await orchestrator._process_listening(frame)
    
    # Since we mocked TTS and LLM, the orchestrator should immediately transition 
    # back to IDLE after it finishes streaming and speaking
    assert orchestrator.state == AgentState.IDLE

    # Verify things were called
    mock_pipeline_dependencies["stt"].transcribe.assert_called()
    mock_pipeline_dependencies["tts"].synthesize.assert_called()
    assert len(mock_pipeline_dependencies["playback"].played_audio) > 0

    orchestrator._running = False
    mock_pipeline_dependencies["capture"].stop()
