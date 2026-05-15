"""Sentence boundary detection for streaming TTS."""

import re
from typing import List, Optional

try:
    import nltk
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

# Regex fallback for sentence splitting when nltk is unavailable.
# Handles common abbreviations to avoid false splits.
# Note: Python's re module does not support variable length lookbehinds, 
# so we avoid complex regex here and use simple splitting in _regex_split.


class SentenceBuffer:
    """Accumulates streaming tokens and emits complete sentences.
    
    Uses NLTK punkt tokenizer for robust sentence boundary detection
    that handles abbreviations, decimals, and other edge cases.
    Falls back to regex-based splitting if NLTK is not installed.
    """

    def __init__(self, min_length: int = 10):
        self._buffer = ""
        self._min_length = min_length
        self._use_nltk = _HAS_NLTK
        if self._use_nltk:
            self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download punkt_tab tokenizer data if not present."""
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                self._use_nltk = False

    def add_token(self, token: str) -> List[str]:
        """Add a token and return any complete sentences.
        
        Returns a list of complete sentences (may be empty if no
        sentence boundary was detected yet).
        """
        self._buffer += token
        return self._extract_sentences()

    def _extract_sentences(self) -> List[str]:
        """Extract complete sentences from the buffer."""
        if len(self._buffer.strip()) < self._min_length:
            return []

        if self._use_nltk:
            sentences = nltk.sent_tokenize(self._buffer.strip())
        else:
            sentences = self._regex_split(self._buffer.strip())

        if len(sentences) <= 1:
            return []

        # All but the last are complete sentences
        complete = []
        for sent in sentences[:-1]:
            sent = sent.strip()
            if len(sent) >= self._min_length:
                complete.append(sent)

        # Keep the last (incomplete) fragment in the buffer
        self._buffer = sentences[-1]
        return complete

    def _regex_split(self, text: str) -> List[str]:
        """Fallback sentence splitter using regex."""
        # Split on sentence-ending punctuation followed by space and uppercase
        parts = re.split(r'([.!?])\s+(?=[A-Z])', text)
        # Re-join punctuation with preceding text
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] in '.!?':
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                sentences.append(parts[i])
                i += 1
        return [s.strip() for s in sentences if s.strip()]

    def flush(self) -> Optional[str]:
        """Flush remaining buffer content as a final sentence.
        
        Call this when the LLM stream ends to get any remaining text.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining and len(remaining) >= 3:  # Lower threshold for final flush
            return remaining
        return None

    def clear(self):
        """Clear the buffer without emitting anything."""
        self._buffer = ""

    @property
    def pending_text(self) -> str:
        """Return current buffer contents (for debugging)."""
        return self._buffer
