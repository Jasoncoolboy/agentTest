"""Live Gemini API smoke test (via OpenAI-compat endpoint).

Loads voice_agent/.env on import so pytest sees GEMINI_API_KEY.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set (add to .env or environment)",
)


@pytest.mark.asyncio
async def test_gemini_stream_smoke():
    from src.llm.client import LLMClient
    from src.llm.history import ConversationHistory

    client = LLMClient(
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=64,
        timeout_seconds=60,
        max_retries=1,
    )
    history = ConversationHistory()
    history.set_system_prompt("You are a test harness. Reply with one very short sentence.")
    history.add_user("Reply with exactly: OK")

    parts: list[str] = []
    async for token in client.stream_response(history, tool_registry=None):
        parts.append(token)

    text = "".join(parts).strip()
    assert len(text) > 0, "Expected non-empty streamed response from Gemini"
