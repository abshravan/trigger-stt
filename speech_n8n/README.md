# speech_n8n — Local Speech-to-Text → n8n Voice Pipeline

A production-ready, fully **local** Speech-to-Text (STT) pipeline. It captures
audio from your microphone, detects speech with Voice Activity Detection (VAD),
transcribes it on-device with [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper),
and POSTs the transcription to an [n8n](https://n8n.io) webhook — then displays
n8n's response and returns to listening.

**No cloud STT APIs are used.** Speech recognition runs entirely on your machine.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [CLI Options](#cli-options)
- [Example n8n Workflow](#example-n8n-workflow)
- [Example Webhook Payload](#example-webhook-payload)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Extending the System](#extending-the-system)
- [Architectural Decisions](#architectural-decisions)
- [Future Enhancements & Trade-offs](#future-enhancements--trade-offs)

---

## Project Overview

```
Microphone
   ↓
(Optional) Wake Word Detection      ← openWakeWord ("Hey Jarvis", "Computer", …)
   ↓
Voice Activity Detection (VAD)      ← Silero VAD
   ↓
Faster-Whisper Speech-to-Text       ← local, model loaded once
   ↓
HTTP POST → n8n Webhook
   ↓
Receive & Display n8n Response
   ↓
Return to Listening State
```

Key properties:

- **Local-first**: transcription never leaves your machine.
- **Modular**: each stage (recording, VAD, STT, webhook, wake word) is an
  independent, swappable component with a clear interface.
- **Resilient**: microphone failures, network errors and unexpected exceptions
  are handled gracefully; the loop recovers instead of crashing.
- **Configurable**: everything is driven by environment variables, validated at
  startup (fail-fast).
- **Extensible**: designed to grow into a full voice assistant (Ollama, Home
  Assistant, local agents, more n8n workflows).

---

## Architecture

```
                         ┌────────────────────────────────────────────┐
                         │                  app.py                     │
                         │              VoiceAgent (loop)              │
                         └───────┬───────────────┬──────────────┬─────┘
                                 │               │              │
            ┌────────────────────┘               │              └──────────────────┐
            ▼                                     ▼                                 ▼
   ┌──────────────────┐               ┌─────────────────────┐            ┌───────────────────┐
   │  AudioRecorder   │  frames       │  WakeWordDetector   │            │     N8nClient     │
   │  (sounddevice)   ├──────────────▶│   (openWakeWord /   │            │  (requests +      │
   │  16 kHz mono     │               │    Null detector)   │            │   retries/backoff)│
   └──────────────────┘               └─────────────────────┘            └───────────────────┘
            │ frames                                                              ▲
            ▼                                                                     │ payload
   ┌──────────────────┐   utterance   ┌─────────────────────┐  result            │
   │ UtteranceCollector├─────────────▶│   WhisperEngine     ├────────────────────┘
   │  + VadEngine      │              │  (faster-whisper,    │
   │  (Silero VAD)     │              │   loaded once)       │
   └──────────────────┘              └─────────────────────┘

   Cross-cutting: config.py (validated Config), utils/logger.py (structured logs),
   utils/audio_utils.py (WAV save, conversions)
```

Each component is constructed once in `VoiceAgent.from_config()` and injected,
keeping global state minimal and making every piece independently testable.

---

## Project Structure

```
speech_n8n/
├── app.py                  # Entry point + VoiceAgent orchestration loop
├── config.py               # Env-driven Config dataclass + validation
├── requirements.txt
├── README.md
├── .env.example
├── pytest.ini
├── stt/
│   ├── whisper_engine.py   # Faster-Whisper wrapper (load once, structured result)
│   └── audio_recorder.py   # sounddevice microphone capture
├── vad/
│   └── vad_engine.py       # Silero VAD + UtteranceCollector
├── webhook/
│   └── n8n_client.py       # HTTP POST with retries/backoff + payload builder
├── wakeword/
│   └── wakeword_engine.py  # openWakeWord detector + Null detector + factory
├── utils/
│   ├── logger.py           # Human-readable / JSON structured logging
│   └── audio_utils.py      # WAV saving, int16/float conversions
└── tests/                  # pytest suite (config, webhook, whisper, vad, utils)
```

---

## Installation

### 1. Prerequisites

- **Python 3.11+**
- A working **microphone**
- **PortAudio** (required by `sounddevice`):
  - macOS: `brew install portaudio`
  - Debian/Ubuntu: `sudo apt-get install -y portaudio19-dev`
  - Windows: bundled with the `sounddevice` wheel (no action needed)
- An **n8n** instance with a Webhook node (local or remote)

### 2. Virtual environment

```bash
cd speech_n8n
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run downloads the Whisper and Silero VAD models automatically and
> caches them locally. `torch` is a sizeable download; on CPU-only machines the
> default `int8` compute type keeps things fast and light.

### 4. Configure

```bash
cp .env.example .env
# edit .env and set N8N_WEBHOOK_URL (and anything else you want to tweak)
```

---

## Configuration

All configuration lives in `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Description |
| --- | --- | --- |
| `N8N_WEBHOOK_URL` | *(required)* | n8n webhook URL to receive transcriptions |
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `DEVICE` | `cpu` | `cpu`/`cuda`/`auto` |
| `COMPUTE_TYPE` | `int8` | `int8`/`int8_float16`/`float16`/`float32` |
| `BEAM_SIZE` | `5` | Whisper decoding beam size |
| `LANGUAGE` | `en` | ISO code, or `auto` to detect |
| `SAMPLE_RATE` | `16000` | Capture/VAD/STT sample rate (Hz) |
| `MIN_SPEECH_DURATION` | `0.25` | Minimum speech (s) for a valid utterance |
| `SILENCE_TIMEOUT` | `1.5` | Trailing silence (s) that ends an utterance |
| `MAX_RECORDING_SECONDS` | `20` | Hard cap on one utterance (s) |
| `VAD_THRESHOLD` | `0.5` | Silero speech-probability threshold |
| `WAKEWORD_ENABLED` | `false` | Enable wake word gating |
| `WAKEWORD_MODELS` | `hey_jarvis` | Comma-separated openWakeWord models |
| `WAKEWORD_THRESHOLD` | `0.5` | Wake word detection threshold |
| `WEBHOOK_TIMEOUT` | `10` | Per-request timeout (s) |
| `WEBHOOK_MAX_RETRIES` | `3` | Extra attempts after first failure |
| `WEBHOOK_BACKOFF_FACTOR` | `0.5` | Exponential backoff base (s) |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |
| `LOG_JSON` | `false` | Emit JSON logs instead of human-readable |
| `SAVE_AUDIO_DIR` | *(empty)* | If set, save each utterance as a WAV here |

Configuration is **validated at startup**. Invalid values cause an immediate,
descriptive error and a non-zero exit code — the app never starts in a bad
state.

---

## Running

```bash
source .venv/bin/activate
python app.py
```

Expected console output:

```
Loading configuration...
Initializing models...
Initializing microphone...
Listening...
Speech detected.
Transcribing...
You said: "turn on the lights"
Sending to n8n...
n8n responded successfully.
n8n response: {'status': 'ok'}
Listening...
```

Stop with `Ctrl+C`.

### Wake word mode

```bash
# via .env
WAKEWORD_ENABLED=true

# or via CLI for a one-off run
python app.py --wakeword
```

The app then idles until it hears a wake word (e.g. *"Hey Jarvis"*) before it
starts capturing an utterance.

---

## CLI Options

```
python app.py [options]

--env-file PATH       Path to the .env file (default: .env)
--model NAME          Override Whisper model (tiny|base|small|medium|large-v3)
--webhook-url URL     Override the n8n webhook URL
--wakeword            Force-enable wake word detection
--json-logs           Emit logs as JSON
--log-level LEVEL     Override the log level
```

Examples:

```bash
python app.py --model small --log-level DEBUG
python app.py --webhook-url https://n8n.example.com/webhook/voice --json-logs
```

---

## Example n8n Workflow

1. Add a **Webhook** node:
   - HTTP Method: `POST`
   - Path: `voice-agent` (→ URL `http://localhost:5678/webhook/voice-agent`)
   - Respond: *Using 'Respond to Webhook' node* (or "Immediately" with last node data)
2. Add nodes to act on `{{$json["text"]}}` (e.g. an **IF**/switch to route
   commands, an HTTP Request to Home Assistant, an LLM node backed by Ollama…).
3. Add a **Respond to Webhook** node returning JSON, e.g.:
   ```json
   { "status": "ok", "reply": "Lights turned on" }
   ```

The app prints whatever JSON (or text) the webhook returns.

---

## Example Webhook Payload

The app POSTs this JSON body:

```json
{
  "text": "Turn on the lights",
  "timestamp": "2026-06-14T12:34:56.789012+00:00",
  "language": "en",
  "confidence": 0.96,
  "source": "desktop_voice_agent"
}
```

---

## Testing

```bash
pip install -r requirements.txt   # or just: pip install pytest numpy requests python-dotenv
pytest
```

The suite covers configuration loading/validation, webhook payload generation,
retry/backoff logic, the Whisper interface (load-once + structured result via a
fake backend), the VAD utterance collector, and audio utilities. Heavy models
(Faster-Whisper, Silero, openWakeWord) are **mocked**, so tests run in well
under a second and need no microphone or GPU.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Failed to initialize microphone` | Ensure a mic is connected and not in use; install PortAudio; check OS mic permissions. |
| `The 'sounddevice' library ... is not available` | `pip install sounddevice` and install PortAudio for your OS. |
| `Configuration error: N8N_WEBHOOK_URL is required` | Set `N8N_WEBHOOK_URL` in `.env`. |
| Slow transcription | Use a smaller model (`--model tiny`/`base`), keep `COMPUTE_TYPE=int8` on CPU, or use a CUDA GPU with `DEVICE=cuda COMPUTE_TYPE=float16`. |
| Speech cut off early | Increase `SILENCE_TIMEOUT`; lower `VAD_THRESHOLD`. |
| Picks up noise as speech | Raise `VAD_THRESHOLD`; raise `MIN_SPEECH_DURATION`. |
| Wake word never fires | Lower `WAKEWORD_THRESHOLD`; confirm the model name in `WAKEWORD_MODELS`. |
| `n8n request failed` | Check the URL, that the workflow is **active**, and n8n is reachable. |
| First run is slow | Models download on first use; subsequent runs use the local cache. |

Enable `LOG_LEVEL=DEBUG` for detailed diagnostics, and set `SAVE_AUDIO_DIR` to
inspect exactly what audio was captured.

---

## Extending the System

The architecture is built for growth. Common extension points:

- **Swap the STT engine**: implement a class with the same
  `load()` / `transcribe(audio) -> TranscriptionResult` interface as
  `WhisperEngine` and inject it in `VoiceAgent.from_config`.
- **Different wake word engine**: implement the `WakeWordDetector` protocol
  (`enabled`, `reset()`, `process(frame) -> Optional[str]`) and return it from
  `wakeword.create_detector`.
- **Post-process / route transcriptions**: the n8n workflow is the natural place
  for routing, LLM calls (Ollama), and Home Assistant control — no code change
  needed.
- **Add text-to-speech**: read n8n's `reply` field and speak it (a `pyttsx3`
  dependency is included for a local TTS option).
- **Multiple webhooks / plugins**: wrap `N8nClient` or add a small dispatcher
  that fans a transcription out to several handlers.

---

## Architectural Decisions

- **Immutable, validated `Config` dataclass.** A single source of truth, parsed
  and validated once at startup (fail-fast). Frozen to discourage hidden global
  mutation.
- **Dependency injection in `VoiceAgent`.** Components are constructed in one
  factory (`from_config`) and passed in, so each can be unit-tested in isolation
  and swapped without touching the loop.
- **Models loaded exactly once.** `WhisperEngine`, `VadEngine` and the wake word
  model each guard their `load()` with an idempotency check; transcription never
  reloads the model — critical for latency and memory.
- **Silero VAD windowing.** Silero requires fixed 512-sample windows at 16 kHz.
  `UtteranceCollector` normalizes arbitrary frames into windows and keeps a
  short **pre-roll** buffer so the first word is never clipped.
- **Wake word as a Protocol with a Null implementation.** Enabling/disabling
  wake word support requires no branching in the core loop — the `Null`
  detector is injected when disabled (Feature 5).
- **Webhook resilience.** `N8nClient` retries only transient failures (network
  errors, HTTP 5xx) with exponential backoff, never retries 4xx, and never
  raises on failure so the loop stays alive.
- **Structured logging with a `component` field.** Human-readable by default,
  JSON on demand — ready for log aggregation.
- **Separation of console UX from logs.** User-facing progress messages go to
  stdout; structured logs go to stderr, so the two never tangle.

---

## Future Enhancements & Trade-offs

Planned / possible improvements:

- **Streaming transcription** (partial results) for lower perceived latency —
  trade-off: more complex buffering and accuracy tuning.
- **Barge-in / full-duplex** so the user can interrupt TTS playback.
- **GPU auto-tuning** (`DEVICE=auto` picking CUDA + `float16`).
- **Metrics endpoint** (Prometheus) for latency/throughput dashboards — a basic
  per-utterance timing is already logged.
- **Plugin manager** to register multiple downstream handlers beyond a single
  webhook.
- **Local LLM (Ollama) pre-routing** before hitting n8n, for intent extraction.
- **Home Assistant** direct integration as an alternative sink to n8n.
- **Push-to-talk / global hotkey** capture (the `pynput` dependency is included
  to make this straightforward to add).

Trade-offs already made:

- Chose **Faster-Whisper** (CTranslate2) over `openai-whisper` for ~4× CPU
  speed and lower memory at `int8`, at the cost of an extra native dependency.
- Chose a **callback-based single persistent audio stream** (lower latency,
  bounded buffering) over re-opening streams per utterance (simpler but slower
  and error-prone).
- Whisper provides no calibrated confidence; we approximate it from segment
  average log-probabilities. It is a useful relative signal, not a probability.

---

## License

Provided as-is for use and extension within your own projects.
