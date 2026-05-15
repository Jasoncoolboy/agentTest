import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class MockLLMClient:
    """Mocks the LLMClient and yields predefined responses."""

    def __init__(self, predefined_responses=None):
        # predefined_responses: list of strings to stream back
        self.predefined_responses = predefined_responses or ["Hello, ", "this ", "is a ", "mock ", "response."]
        self.call_count = 0

    async def stream_response(self, history, tool_registry=None) -> AsyncGenerator[str, None]:
        # Return the next predefined response based on call count
        if self.call_count < len(self.predefined_responses):
            response_text = self.predefined_responses[self.call_count]
        else:
            response_text = "I am out of predefined mock responses."
            
        self.call_count += 1
        
        # Split by spaces to simulate token streaming
        words = response_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05) # simulate network latency
