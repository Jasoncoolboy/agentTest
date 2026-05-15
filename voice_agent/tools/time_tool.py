"""Time tool - returns current date and time."""

from datetime import datetime

from tools.base import BaseTool


class TimeTool(BaseTool):
    """Returns the current local date and time."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Get the current local date and time. Use when the user asks what time or date it is."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format: 'full' for date and time, 'time' for time only, 'date' for date only",
                    "enum": ["full", "time", "date"],
                    "default": "full",
                },
            },
            "required": [],
        }

    async def execute(self, format: str = "full") -> str:
        now = datetime.now()

        if format == "time":
            return now.strftime("The current time is %I:%M %p")
        elif format == "date":
            return now.strftime("Today is %A, %B %d, %Y")
        else:
            return now.strftime("It's %A, %B %d, %Y at %I:%M %p")
