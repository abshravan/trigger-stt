"""Application entry point for the local STT -> n8n voice pipeline.

Pipeline:
    Microphone -> (optional) Wake Word -> VAD -> Faster-Whisper -> n8n webhook
    -> display response -> back to listening.

Run with::

    python app.py

See ``python app.py --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from config import Config, ConfigError
from stt.audio_recorder import AudioRecorder, MicrophoneError
from stt.whisper_engine import WhisperEngine
from utils.audio_utils import save_wav
from utils.logger import configure_logging, get_logger
from vad.vad_engine import UtteranceCollector, VadEngine
from wakeword.wakeword_engine import WakeWordDetector, create_detector
from webhook.n8n_client import N8nClient, build_payload

logger = get_logger("app")


def console(message: str) -> None:
    """Print a user-facing status message to stdout (separate from logs)."""
    print(message, flush=True)


class VoiceAgent:
    """Orchestrates the end-to-end voice pipeline.

    Dependencies are injected so the agent is easy to test and extend. Use
    :meth:`from_config` for the standard wiring.
    """

    def __init__(
        self,
        config: Config,
        recorder: AudioRecorder,
        vad: VadEngine,
        collector: UtteranceCollector,
        whisper: WhisperEngine,
        wakeword: WakeWordDetector,
        client: N8nClient,
    ) -> None:
        self.config = config
        self.recorder = recorder
        self.vad = vad
        self.collector = collector
        self.whisper = whisper
        self.wakeword = wakeword
        self.client = client
        self._running = False

    @classmethod
    def from_config(cls, config: Config) -> "VoiceAgent":
        """Build a fully-wired :class:`VoiceAgent` from configuration."""
        recorder = AudioRecorder(sample_rate=config.sample_rate)
        vad = VadEngine(
            sample_rate=config.sample_rate, threshold=config.vad_threshold
        )
        collector = UtteranceCollector(
            vad=vad,
            sample_rate=config.sample_rate,
            silence_timeout=config.silence_timeout,
            min_speech_duration=config.min_speech_duration,
            max_recording_seconds=config.max_recording_seconds,
        )
        whisper = WhisperEngine(
            model_size=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            beam_size=config.beam_size,
            language=config.whisper_language,
        )
        wakeword = create_detector(
            enabled=config.wakeword_enabled,
            models=config.wakeword_models,
            threshold=config.wakeword_threshold,
            sample_rate=config.sample_rate,
        )
        client = N8nClient(
            webhook_url=config.n8n_webhook_url,
            timeout=config.webhook_timeout,
            max_retries=config.webhook_max_retries,
            backoff_factor=config.webhook_backoff_factor,
        )
        return cls(
            config=config,
            recorder=recorder,
            vad=vad,
            collector=collector,
            whisper=whisper,
            wakeword=wakeword,
            client=client,
        )

    # -- startup ----------------------------------------------------------
    def initialize(self) -> None:
        """Load models and open the microphone (each loaded exactly once)."""
        console("Initializing models...")
        self.whisper.load()
        self.vad.load()
        if self.wakeword.enabled and hasattr(self.wakeword, "load"):
            self.wakeword.load()  # type: ignore[attr-defined]

        console("Initializing microphone...")
        self.recorder.start()

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        """Run the continuous listen -> transcribe -> send loop."""
        self._running = True
        if self.wakeword.enabled:
            console(
                "Listening for wake word "
                f"({', '.join(self.config.wakeword_models)})..."
            )
        else:
            console("Listening...")

        while self._running:
            try:
                self._listen_once()
            except MicrophoneError as exc:
                logger.error("Microphone error: %s", exc)
                console(f"Microphone error: {exc}")
                if not self._recover_microphone():
                    break
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.exception("Unexpected error in main loop: %s", exc)
                console(f"Recovered from an error: {exc}")
                time.sleep(0.5)

    def _listen_once(self) -> None:
        """Handle a single wake-word -> utterance -> webhook cycle."""
        if self.wakeword.enabled:
            if not self._await_wake_word():
                return
            console("Wake word detected. Listening...")

        utterance = self.collector.collect(lambda: self.recorder.read_frame(timeout=1.0))
        if utterance is None:
            return

        console("Speech detected.")
        logger.info(
            "Captured utterance: %.2fs (ended by %s).",
            utterance.duration,
            utterance.reason,
        )

        if self.config.save_audio_dir:
            path = save_wav(
                utterance.audio, self.config.sample_rate, self.config.save_audio_dir
            )
            if path:
                logger.debug("Saved utterance to %s", path)

        console("Transcribing...")
        result = self.whisper.transcribe(utterance.audio)
        if not result.transcribed_text:
            console("No speech recognized. Listening...")
            return

        console(f'You said: "{result.transcribed_text}"')
        self._send_to_n8n(result)

        if self.wakeword.enabled:
            self.wakeword.reset()
            console(
                "Listening for wake word "
                f"({', '.join(self.config.wakeword_models)})..."
            )
        else:
            console("Listening...")

    def _await_wake_word(self) -> bool:
        """Block until the wake word fires; return ``False`` if stopped."""
        while self._running:
            frame = self.recorder.read_frame(timeout=1.0)
            if frame is None:
                continue
            if self.wakeword.process(frame) is not None:
                return True
        return False

    def _send_to_n8n(self, result) -> None:  # noqa: ANN001 - TranscriptionResult
        """Post a transcription to n8n and display the response."""
        console("Sending to n8n...")
        payload = build_payload(
            text=result.transcribed_text,
            language=result.detected_language,
            confidence=result.confidence_score,
        )
        response = self.client.send(payload)
        if response.ok:
            console("n8n responded successfully.")
            console(f"n8n response: {response.body}")
        else:
            console(f"n8n request failed: {response.error}")

    def _recover_microphone(self) -> bool:
        """Attempt to restart the microphone after a failure."""
        console("Attempting to recover microphone...")
        for attempt in range(1, 4):
            time.sleep(min(2 ** attempt, 8))
            try:
                self.recorder.stop()
                self.recorder.start()
                console("Microphone recovered. Listening...")
                return True
            except MicrophoneError as exc:
                logger.warning("Recovery attempt %d failed: %s", attempt, exc)
        console("Could not recover microphone. Exiting.")
        return False

    # -- shutdown ---------------------------------------------------------
    def stop(self) -> None:
        """Signal the loop to stop and release resources."""
        self._running = False

    def shutdown(self) -> None:
        """Release all resources."""
        self.recorder.stop()
        self.client.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (bonus: CLI overrides)."""
    parser = argparse.ArgumentParser(
        description="Local Speech-to-Text pipeline that sends transcriptions "
        "to an n8n webhook."
    )
    parser.add_argument("--env-file", default=".env", help="Path to the .env file.")
    parser.add_argument(
        "--model",
        dest="whisper_model",
        help="Override the Whisper model size (tiny|base|small|medium|large-v3).",
    )
    parser.add_argument(
        "--webhook-url", dest="n8n_webhook_url", help="Override the n8n webhook URL."
    )
    parser.add_argument(
        "--wakeword",
        dest="wakeword",
        action="store_true",
        help="Enable wake word detection regardless of the .env setting.",
    )
    parser.add_argument(
        "--json-logs", action="store_true", help="Emit logs as JSON."
    )
    parser.add_argument("--log-level", help="Override the log level.")
    return parser.parse_args(argv)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Return a new Config with CLI overrides applied and re-validated."""
    import dataclasses

    overrides = {}
    if args.whisper_model:
        overrides["whisper_model"] = args.whisper_model
    if args.n8n_webhook_url:
        overrides["n8n_webhook_url"] = args.n8n_webhook_url
    if args.wakeword:
        overrides["wakeword_enabled"] = True
    if args.json_logs:
        overrides["log_json"] = True
    if args.log_level:
        overrides["log_level"] = args.log_level.upper()

    if not overrides:
        return config
    updated = dataclasses.replace(config, **overrides)
    updated.validate()
    return updated


def main(argv: Optional[list[str]] = None) -> int:
    """Program entry point. Returns a process exit code."""
    args = parse_args(argv)

    console("Loading configuration...")
    try:
        config = Config.from_env(env_file=args.env_file)
        config = _apply_overrides(config, args)
    except ConfigError as exc:
        # Logging may not be configured yet; print directly.
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    configure_logging(level=config.log_level, json_output=config.log_json)
    logger.info("Application starting up.")
    logger.info(
        "Config loaded: model=%s device=%s wakeword=%s webhook=%s",
        config.whisper_model,
        config.device,
        config.wakeword_enabled,
        config.n8n_webhook_url,
    )

    agent = VoiceAgent.from_config(config)
    try:
        agent.initialize()
    except MicrophoneError as exc:
        logger.error("Startup failed: %s", exc)
        console(f"Failed to start: {exc}")
        return 1

    try:
        agent.run()
    except KeyboardInterrupt:
        console("\nStopping (Ctrl+C). Goodbye!")
    finally:
        agent.stop()
        agent.shutdown()
        logger.info("Application shut down cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
