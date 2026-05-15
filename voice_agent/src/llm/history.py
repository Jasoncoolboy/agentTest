"""Conversation history management with token budget.

Maintains chat history and prunes old messages when the
token budget is exceeded.
"""

import logging
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manages conversation messages with automatic token-budget pruning."""

    def __init__(self, max_tokens: int = 3000, max_turns: int = 20):
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._messages: List[Dict[str, str]] = []
        self._system_prompt: Optional[str] = None
        self._encoder = None
        if tiktoken is not None:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("tiktoken encoding failed, using rough token estimation")
        else:
            logger.warning("tiktoken not installed, using rough token estimation")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt (always retained, never pruned)."""
        self._system_prompt = prompt

    def add_user(self, text: str):
        """Add a user message."""
        self._messages.append({"role": "user", "content": text})
        self._prune()

    def add_assistant(self, text: str):
        """Add an assistant response."""
        self._messages.append({"role": "assistant", "content": text})
        self._prune()

    def add_tool_call(self, tool_call_id: str, name: str, arguments: str):
        """Add an assistant message with a tool call."""
        self._messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        })

    def add_tool_result(self, tool_call_id: str, content: str):
        """Add a tool result message."""
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._prune()

    def get_messages(self) -> List[Dict]:
        """Get full message list for API call (system + history)."""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        return messages

    def clear(self):
        """Clear all history (keeps system prompt)."""
        self._messages.clear()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Rough fallback: ~4 chars per token
        return len(text) // 4

    def _total_tokens(self) -> int:
        """Count total tokens across all messages."""
        total = 0
        if self._system_prompt:
            total += self._count_tokens(self._system_prompt)
        for msg in self._messages:
            content = msg.get("content") or ""
            total += self._count_tokens(content) + 4  # overhead per message
        return total

    def _prune(self):
        """Remove oldest messages to stay within budget."""
        # Prune by turn count
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)

        # Prune by token budget
        while self._total_tokens() > self.max_tokens and len(self._messages) > 2:
            # Remove oldest message (try to remove in pairs)
            self._messages.pop(0)
            logger.debug("Pruned oldest message from history")

    @property
    def turn_count(self) -> int:
        """Number of user/assistant turn pairs."""
        return sum(1 for m in self._messages if m["role"] == "user")
