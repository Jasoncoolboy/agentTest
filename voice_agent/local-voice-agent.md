# Local AI Voice Agent - Implementation Plan

## Context

Build a fully functional AI voice agent from scratch that runs on a resource-constrained laptop (8GB RAM, no GPU). The agent provides a complete voice pipeline: always-on wake word detection, voice activity detection, speech-to-text, streaming LLM responses with tool calling, and real-time text-to-speech playback with barge-in support.

---

## Tech Stack

| Component | Selection | Rationale |
|-----------|-----------|-----------|
| Wake Word | OpenWakeWord | Open-source, free, ~50MB RAM |
| VAD | Silero VAD | Lightweight, accurate, open-source |
| STT | faster-whisper (distil-small.en, int8) | ~150MB RAM, 1-3s latency on CPU |
| LLM | Google Gemini API (streaming, OpenAI-compatible) | No local GPU, supports tool calling |
| TTS | Piper TTS (en_US-lessac-medium) | ~80MB, 50-150ms per sentence on CPU |
| Audio I/O | sounddevice | Cross-platform, low-level PortAudio binding |
| Orchestration | asyncio event loop | Non-blocking, concurrent task coordination |

**Total estimated RAM:** ~585MB peak (leaves ample headroom on 8GB)

---

## Project Structure

```
voice_agent/
├── config/
│   └── default.yaml              # All tunable parameters
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── orchestrator.py           # Async pipeline state machine
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py            # Mic input (16kHz, float32, 30ms frames)
│   │   └── playback.py           # Speaker output (22050Hz)
│   ├── wake_word/
│   │   ├── __init__.py
│   │   └── detector.py           # OpenWakeWord wrapper
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero.py             # Silero VAD wrapper
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper.py            # faster-whisper wrapper
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py             # Gemini streaming client (OpenAI compat) + tool calls
│   │   ├── tools.py              # Tool registry and schema generation
│   │   └── history.py            # Conversation history with token budget
│   ├── tts/
│   │   ├── __init__.py
│   │   └── piper.py              # Piper TTS wrapper
│   └── utils/
│       ├── __init__.py
│       ├── sentence_buffer.py    # NLTK-based sentence boundary detection
│       └── exceptions.py         # Custom exception types
├── tools/
│   ├── __init__.py
│   ├── base.py                   # Abstract tool interface
│   ├── weather.py                # Example: weather tool (Open-Meteo)
│   └── time_tool.py              # Example: current time/date
├── models/                       # Downloaded model files (gitignored)
├── scripts/
│   ├── download_models.py        # One-shot model downloader
│   └── test_microphone.py        # Audio device sanity check
├── requirements.txt
├── .env.example                  # GEMINI_API_KEY placeholder
└── .gitignore
```

---

## Pipeline State Machine

```
IDLE (wake word listening)
  │ wake word detected
  ▼
LISTENING (VAD active, accumulating audio)
  │ silence timeout (700ms)
  ▼
PROCESSING (STT → LLM streaming → sentence buffering)
  │ first sentence ready
  ▼
SPEAKING (TTS playback, streaming continues)
  │ barge-in detected OR TTS finished
  ▼
  └── back to IDLE (or LISTENING on barge-in)
```

---

## Key Implementation Details

### Async Architecture
- `asyncio` event loop with concurrent tasks for audio capture, wake word, VAD, and playback
- Blocking operations (faster-whisper, Piper synthesis) run in `ThreadPoolExecutor(max_workers=2)`
- LLM streaming uses async HTTP (OpenAI client's native async support)
- Audio frames distributed to consumers via `asyncio.Queue`
- State transitions coordinated via `asyncio.Event`

### Barge-in (gap-based mode)
- During SPEAKING, VAD monitors mic input in brief pauses between sentences
- If speech detected (>150ms sustained, probability >0.6): stop playback, clear TTS queue, cancel LLM stream, transition to LISTENING
- Config supports modes: "gap" (default), "continuous", "disabled"
- Headphones recommended to avoid TTS-to-mic feedback

### Sentence Buffering
- NLTK `punkt_tab` tokenizer for robust sentence boundary detection
- Handles abbreviations, decimals, ellipsis correctly
- Emits complete sentences to TTS as soon as detected (streaming playback)
- Minimum sentence length threshold (10 chars) to avoid tiny fragments

### Tool Calling Flow
1. LLM stream produces `tool_calls` delta chunks (OpenAI-compatible format)
2. Accumulate JSON arguments until stream finishes with `finish_reason="tool_calls"`
3. Execute tool async (with 10s timeout)
4. Append result to history, make follow-up API call
5. Follow-up stream produces natural language response → sentence buffer → TTS

### Conversation History
- Token budget: 3000 tokens max (tracked via tiktoken)
- Pruning: remove oldest user/assistant pairs when budget exceeded
- System prompt always retained
- Max 20 turns before forced pruning

### Lazy Model Loading (RAM Conservation)
1. Startup: load OpenWakeWord + Silero VAD only (~55MB)
2. First wake detection: load faster-whisper (~150MB, one-time 1-2s delay)
3. First TTS request: load Piper model (~80MB, ~100ms)

### Error Handling
- Empty STT transcription → return to IDLE silently (3 consecutive → "I'm listening" prompt)
- API timeout (30s) → speak "I'm having trouble connecting" → IDLE
- Tool failure → pass error to LLM, it responds naturally
- Audio device error → list available devices, attempt re-open once

---

## Dependencies (requirements.txt)

```
sounddevice>=0.4.6
numpy>=1.24
openwakeword>=0.6.0
torch>=2.1 (CPU-only: --index-url https://download.pytorch.org/whl/cpu)
faster-whisper>=1.0.0
openai>=1.30
piper-tts>=1.2.0
pyyaml>=6.0
nltk>=3.8
tiktoken>=0.7
python-dotenv>=1.0
```

---

## Implementation Order

1. Project scaffolding (directories, config, requirements, .gitignore, .env)
2. `scripts/download_models.py` — model downloader
3. `src/audio/capture.py` + `scripts/test_microphone.py` — verify audio works
4. `src/audio/playback.py` — audio output
5. `src/wake_word/detector.py` — wake word detection loop
6. `src/vad/silero.py` — voice activity detection
7. `src/stt/whisper.py` — speech-to-text
8. `src/utils/sentence_buffer.py` — sentence boundary detection
9. `src/llm/history.py` — conversation history management
10. `src/llm/client.py` — Gemini streaming client
11. `src/llm/tools.py` + `tools/` — tool calling framework + examples
12. `src/tts/piper.py` — text-to-speech synthesis
13. `src/orchestrator.py` — state machine tying everything together
14. `src/main.py` — entry point with startup sequence
15. Barge-in logic integration
16. End-to-end testing and tuning

---

## Verification

1. Run `scripts/test_microphone.py` — confirm audio capture/playback works
2. Run `scripts/download_models.py` — confirm all models download successfully
3. Run `python -m src.main` — confirm startup without errors
4. Say wake word — confirm detection logged in console
5. Speak a question — confirm transcription printed, LLM response streamed, TTS plays
6. Ask "what time is it?" — confirm tool calling works and response is spoken
7. Interrupt during response — confirm barge-in stops playback (if mode != disabled)
8. Monitor RAM usage with Task Manager — confirm stays under ~1GB

---

## Windows-Specific Notes

- Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Piper: use pre-built Windows binary (piper.exe) via subprocess
- asyncio: may need `WindowsSelectorEventLoopPolicy` or `ProactorEventLoop`
- File paths: use `pathlib.Path` throughout
- Audio devices: expose device index in config for manual selection
