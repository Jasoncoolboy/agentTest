"""Custom exception types for the voice agent pipeline."""


class VoiceAgentError(Exception):
    """Base exception for voice agent errors."""
    pass


class AudioDeviceError(VoiceAgentError):
    """Raised when audio device cannot be opened or fails during operation."""
    pass


class WakeWordError(VoiceAgentError):
    """Raised when wake word model fails to load or process."""
    pass


class VADError(VoiceAgentError):
    """Raised when VAD model fails."""
    pass


class STTError(VoiceAgentError):
    """Raised when speech-to-text transcription fails."""
    pass


class LLMError(VoiceAgentError):
    """Raised when LLM API call fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM API call times out."""
    pass


class TTSError(VoiceAgentError):
    """Raised when text-to-speech synthesis fails."""
    pass


class ModelNotFoundError(VoiceAgentError):
    """Raised when a required model file is missing."""
    pass
