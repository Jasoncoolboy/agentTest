"""Tool calling framework for the voice agent.

Provides a registry of tools that can be called by the LLM
via function calling, with JSON schema generation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool function name (used in API calls)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for the tool's parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return a result string."""
        ...


class ToolRegistry:
    """Registry of available tools with schema generation."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict]:
        """Generate OpenAI-compatible function schemas for all tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas

    async def execute(self, name: str, arguments: Dict[str, Any], timeout: float = 10.0) -> str:
        """Execute a tool by name with the given arguments.
        
        Args:
            name: Tool function name
            arguments: Parsed arguments dict
            timeout: Maximum execution time in seconds
            
        Returns:
            Tool result as a string
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"

        try:
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=timeout,
            )
            logger.info(f"Tool '{name}' executed successfully")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool '{name}' timed out after {timeout}s")
            return f"Error: Tool '{name}' timed out"
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return f"Error executing '{name}': {str(e)}"

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def has_tools(self) -> bool:
        return len(self._tools) > 0
