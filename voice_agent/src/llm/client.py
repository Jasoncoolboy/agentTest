"""DeepSeek LLM streaming client with tool calling support.

Uses the OpenAI-compatible API with streaming responses and
function calling capabilities.
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from src.llm.history import ConversationHistory
from src.llm.tools import ToolRegistry
from src.utils.exceptions import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)


class LLMClient:
    """Streaming LLM client for DeepSeek with tool calling."""

    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout_seconds: int = 30,
        max_retries: int = 1,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise LLMError(
                f"Environment variable '{api_key_env}' not set. "
                "Copy .env.example to .env and add your API key."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def stream_response(
        self,
        history: ConversationHistory,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens, handling tool calls transparently.
        
        Yields text tokens as they arrive. If the LLM decides to call a tool,
        this method handles the tool execution and makes a follow-up API call,
        then yields the follow-up response tokens.
        
        Args:
            history: Conversation history manager
            tool_registry: Optional tool registry for function calling
            
        Yields:
            Text tokens (strings) as they stream in
        """
        try:
            async for token in self._stream_with_tools(history, tool_registry):
                yield token
        except asyncio.TimeoutError:
            raise LLMTimeoutError("LLM API request timed out")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {e}") from e

    async def _stream_with_tools(
        self,
        history: ConversationHistory,
        tool_registry: Optional[ToolRegistry],
    ) -> AsyncGenerator[str, None]:
        """Internal: stream with tool call detection and execution."""
        messages = history.get_messages()

        # Prepare request kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        # Add tools if registry has any
        if tool_registry and tool_registry.has_tools:
            kwargs["tools"] = tool_registry.get_schemas()

        # Start streaming
        stream = await self._client.chat.completions.create(**kwargs)

        # Track potential tool calls
        content_buffer = ""
        tool_calls_buffer: Dict[int, Dict] = {}
        has_tool_calls = False

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Handle text content
            if delta.content:
                content_buffer += delta.content
                yield delta.content

            # Handle tool calls (streamed as deltas)
            if delta.tool_calls:
                has_tool_calls = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_buffer[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_buffer[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

        # If content was generated, add to history
        if content_buffer and not has_tool_calls:
            history.add_assistant(content_buffer)
            return

        # If tool calls were made, execute them and make follow-up call
        if has_tool_calls and tool_registry:
            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                tool_name = tc["name"]
                tool_args_str = tc["arguments"]
                tool_call_id = tc["id"]

                logger.info(f"Tool call: {tool_name}({tool_args_str})")

                # Parse arguments
                try:
                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                except json.JSONDecodeError:
                    tool_args = {}

                # Add tool call to history
                history.add_tool_call(tool_call_id, tool_name, tool_args_str)

                # Execute tool
                result = await tool_registry.execute(tool_name, tool_args)
                logger.info(f"Tool result: {result[:100]}")

                # Add result to history
                history.add_tool_result(tool_call_id, result)

            # Make follow-up call with tool results
            follow_up_messages = history.get_messages()
            follow_up_kwargs = {
                "model": self.model,
                "messages": follow_up_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }

            follow_up_stream = await self._client.chat.completions.create(**follow_up_kwargs)
            follow_up_content = ""

            async for chunk in follow_up_stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    follow_up_content += delta.content
                    yield delta.content

            if follow_up_content:
                history.add_assistant(follow_up_content)
