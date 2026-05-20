import pytest

@pytest.fixture
def mock_config():
    return {
        "audio": {
            "sample_rate": 16000,
            "block_size": 480,
            "playback_sample_rate": 22050,
            "device": None,
            "playback_device": None,
        },
        "wake_word": {
            "model_name": "hey_jarvis",
            "threshold": 0.5,
            "debounce_seconds": 1.0,
        },
        "vad": {
            "model_path": "silero_vad.jit",
            "speech_threshold": 0.5,
            "silence_timeout_ms": 700,
            "min_speech_duration_ms": 150,
        },
        "barge_in": {
            "mode": "gap",
            "speech_threshold": 0.6,
            "trigger_duration_ms": 150,
        },
        "stt": {
            "model": "distil-small.en",
            "compute_type": "int8",
            "beam_size": 1,
            "language": "en",
            "vad_filter": True,
        },
        "llm": {
            "api_key_env": "GEMINI_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 512,
            "timeout_seconds": 30,
            "max_retries": 1,
            "system_prompt": "You are a helpful assistant.",
        },
        "history": {
            "max_tokens": 3000,
            "max_turns": 20,
        },
        "tts": {
            "model": "en_US-lessac-medium",
            "model_path": "en_US-lessac-medium.onnx",
        },
        "pipeline": {
            "sentence_min_length": 10,
        }
    }
