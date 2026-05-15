"""Voice Agent - Main Entry Point.

Starts the async voice agent pipeline with all components initialized.

Usage:
    python -m src.main
    
Environment:
    DEEPSEEK_API_KEY: Your DeepSeek API key (required)
"""

import asyncio
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from tools.time_tool import TimeTool
from tools.weather import WeatherTool


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def check_prerequisites(config: dict) -> bool:
    """Verify all prerequisites are met before starting."""
    import os

    errors = []

    # Check API key
    api_key_env = config["llm"]["api_key_env"]
    if not os.getenv(api_key_env):
        errors.append(
            f"Environment variable '{api_key_env}' not set.\n"
            f"  Copy .env.example to .env and add your DeepSeek API key."
        )

    # Check model files
    vad_model = Path(config["vad"]["model_path"])
    if not vad_model.exists():
        errors.append(
            f"VAD model not found: {vad_model}\n"
            f"  Run: python scripts/download_models.py"
        )

    tts_model = Path(config["tts"]["model_path"]) / f"{config['tts']['model']}.onnx"
    if not tts_model.exists():
        errors.append(
            f"TTS model not found: {tts_model}\n"
            f"  Run: python scripts/download_models.py"
        )

    if errors:
        print("\n" + "=" * 50)
        print("PREREQUISITES CHECK FAILED")
        print("=" * 50)
        for err in errors:
            print(f"\n  - {err}")
        print("\n" + "=" * 50)
        return False

    return True


def print_banner(config: dict):
    """Print startup banner."""
    print()
    print("=" * 50)
    print("  Voice Agent - Local AI Assistant")
    print("=" * 50)
    print(f"  Wake word:  '{config['wake_word']['model_name']}'")
    print(f"  STT model:  {config['stt']['model']} ({config['stt']['compute_type']})")
    print(f"  LLM:        {config['llm']['model']}")
    print(f"  TTS:        {config['tts']['model']}")
    print(f"  Barge-in:   {config['barge_in']['mode']}")
    print("=" * 50)
    print(f"  Say '{config['wake_word']['model_name'].replace('_', ' ')}' to activate")
    print(f"  Press Ctrl+C to exit")
    print("=" * 50)
    print()


async def main():
    """Main async entry point."""
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Load config
    config = load_config()

    # Setup logging
    setup_logging(config["pipeline"]["log_level"])
    logger = logging.getLogger(__name__)

    # Check prerequisites
    if not check_prerequisites(config):
        sys.exit(1)

    # Print banner
    print_banner(config)

    # Create orchestrator
    orchestrator = Orchestrator(config)

    # Register tools
    orchestrator.register_tool(TimeTool())
    orchestrator.register_tool(WeatherTool())

    logger.info("Tools registered: " + ", ".join(orchestrator._tool_registry.tool_names))

    # Start the agent
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        await orchestrator.stop()


def run():
    """Entry point handling Windows event loop policy."""
    if sys.platform == "win32":
        # Use SelectorEventLoop on Windows for better compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    run()
