"""Weather tool - fetches weather from Open-Meteo (free, no API key)."""

import json
import urllib.request
import urllib.parse
from typing import Optional

from tools.base import BaseTool


class WeatherTool(BaseTool):
    """Gets current weather for a location using the free Open-Meteo API."""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Get the current weather for a given city or location. "
            "Returns temperature, conditions, and wind speed."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location (e.g., 'London', 'New York', 'Tokyo')",
                },
            },
            "required": ["location"],
        }

    async def execute(self, location: str) -> str:
        try:
            # Step 1: Geocode the location
            coords = await self._geocode(location)
            if coords is None:
                return f"Could not find location: {location}"

            lat, lon, name = coords

            # Step 2: Get weather
            weather = await self._get_weather(lat, lon)
            if weather is None:
                return f"Could not fetch weather for {name}"

            return (
                f"Weather in {name}: {weather['condition']}, "
                f"{weather['temperature']}°C (feels like {weather['feels_like']}°C), "
                f"wind {weather['wind_speed']} km/h, humidity {weather['humidity']}%"
            )
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"

    async def _geocode(self, location: str) -> Optional[tuple]:
        """Get coordinates for a location name."""
        url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": location, "count": 1})
        )

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())

            results = data.get("results", [])
            if not results:
                return None

            r = results[0]
            return (r["latitude"], r["longitude"], r.get("name", location))
        except Exception:
            return None

    async def _get_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Fetch current weather from Open-Meteo."""
        params = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        })
        url = f"https://api.open-meteo.com/v1/forecast?{params}"

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())

            current = data.get("current", {})
            return {
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "condition": self._weather_code_to_text(current.get("weather_code", 0)),
            }
        except Exception:
            return None

    def _weather_code_to_text(self, code: int) -> str:
        """Convert WMO weather code to readable text."""
        codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }
        return codes.get(code, "Unknown conditions")
