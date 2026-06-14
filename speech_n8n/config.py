"""Application configuration.

Configuration is sourced from environment variables (optionally loaded from a
``.env`` file) and exposed through the immutable :class:`Config` dataclass.

The module follows a *fail-fast* philosophy: :meth:`Config.from_env` validates
every value at startup and raises :class:`ConfigError` with an actionable
message when something is wrong, so the application never starts in a
half-configured state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Models supported by Faster-Whisper that we explicitly allow.
SUPPORTED_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
SUPPORTED_DEVICES = ("cpu", "cuda", "auto")
SUPPORTED_COMPUTE_TYPES = ("int8", "int8_float16", "float16", "float32")
SUPPORTED_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable in a forgiving way."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got '{raw}'") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got '{raw}'") from exc


@dataclass(frozen=True)
class Config:
    """Immutable, validated application configuration.

    Instances are created via :meth:`from_env`; the constructor is kept simple
    so it can also be used directly in tests with explicit values.
    """

    # n8n webhook
    n8n_webhook_url: str

    # Whisper / STT
    whisper_model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str = "en"

    # Voice Activity Detection
    sample_rate: int = 16000
    min_speech_duration: float = 0.25
    silence_timeout: float = 1.5
    max_recording_seconds: float = 20.0
    vad_threshold: float = 0.5

    # Wake word
    wakeword_enabled: bool = False
    wakeword_models: List[str] = field(default_factory=lambda: ["hey_jarvis"])
    wakeword_threshold: float = 0.5

    # Webhook networking
    webhook_timeout: float = 10.0
    webhook_max_retries: int = 3
    webhook_backoff_factor: float = 0.5

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Debugging
    save_audio_dir: str = ""

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Config":
        """Build a :class:`Config` from environment variables.

        Args:
            env_file: Path to a ``.env`` file to load before reading variables.
                Pass ``None`` to skip loading a file (useful in tests).

        Returns:
            A validated :class:`Config` instance.

        Raises:
            ConfigError: If any value is missing or invalid.
        """
        if env_file:
            load_dotenv(env_file, override=False)

        language = (os.getenv("LANGUAGE", "en") or "en").strip()

        config = cls(
            n8n_webhook_url=(os.getenv("N8N_WEBHOOK_URL", "") or "").strip(),
            whisper_model=(os.getenv("WHISPER_MODEL", "base") or "base").strip(),
            device=(os.getenv("DEVICE", "cpu") or "cpu").strip().lower(),
            compute_type=(os.getenv("COMPUTE_TYPE", "int8") or "int8").strip(),
            beam_size=_get_int("BEAM_SIZE", 5),
            language=language,
            sample_rate=_get_int("SAMPLE_RATE", 16000),
            min_speech_duration=_get_float("MIN_SPEECH_DURATION", 0.25),
            silence_timeout=_get_float("SILENCE_TIMEOUT", 1.5),
            max_recording_seconds=_get_float("MAX_RECORDING_SECONDS", 20.0),
            vad_threshold=_get_float("VAD_THRESHOLD", 0.5),
            wakeword_enabled=_get_bool("WAKEWORD_ENABLED", False),
            wakeword_models=_parse_csv(os.getenv("WAKEWORD_MODELS", "hey_jarvis")),
            wakeword_threshold=_get_float("WAKEWORD_THRESHOLD", 0.5),
            webhook_timeout=_get_float("WEBHOOK_TIMEOUT", 10.0),
            webhook_max_retries=_get_int("WEBHOOK_MAX_RETRIES", 3),
            webhook_backoff_factor=_get_float("WEBHOOK_BACKOFF_FACTOR", 0.5),
            log_level=(os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper(),
            log_json=_get_bool("LOG_JSON", False),
            save_audio_dir=(os.getenv("SAVE_AUDIO_DIR", "") or "").strip(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate the configuration, raising :class:`ConfigError` on failure."""
        errors: List[str] = []

        if not self.n8n_webhook_url:
            errors.append(
                "N8N_WEBHOOK_URL is required. Set it in your .env file "
                "(see .env.example)."
            )
        elif not self.n8n_webhook_url.startswith(("http://", "https://")):
            errors.append(
                f"N8N_WEBHOOK_URL must start with http:// or https:// "
                f"(got '{self.n8n_webhook_url}')."
            )

        if self.whisper_model not in SUPPORTED_WHISPER_MODELS:
            errors.append(
                f"WHISPER_MODEL must be one of {SUPPORTED_WHISPER_MODELS}, "
                f"got '{self.whisper_model}'."
            )

        if self.device not in SUPPORTED_DEVICES:
            errors.append(
                f"DEVICE must be one of {SUPPORTED_DEVICES}, got '{self.device}'."
            )

        if self.compute_type not in SUPPORTED_COMPUTE_TYPES:
            errors.append(
                f"COMPUTE_TYPE must be one of {SUPPORTED_COMPUTE_TYPES}, "
                f"got '{self.compute_type}'."
            )

        if self.beam_size < 1:
            errors.append(f"BEAM_SIZE must be >= 1, got {self.beam_size}.")

        if self.sample_rate <= 0:
            errors.append(f"SAMPLE_RATE must be positive, got {self.sample_rate}.")

        if self.min_speech_duration < 0:
            errors.append("MIN_SPEECH_DURATION must be >= 0.")

        if self.silence_timeout <= 0:
            errors.append("SILENCE_TIMEOUT must be > 0.")

        if self.max_recording_seconds <= 0:
            errors.append("MAX_RECORDING_SECONDS must be > 0.")

        if not 0.0 <= self.vad_threshold <= 1.0:
            errors.append("VAD_THRESHOLD must be between 0.0 and 1.0.")

        if self.wakeword_enabled and not self.wakeword_models:
            errors.append(
                "WAKEWORD_ENABLED is true but WAKEWORD_MODELS is empty."
            )

        if not 0.0 <= self.wakeword_threshold <= 1.0:
            errors.append("WAKEWORD_THRESHOLD must be between 0.0 and 1.0.")

        if self.webhook_timeout <= 0:
            errors.append("WEBHOOK_TIMEOUT must be > 0.")

        if self.webhook_max_retries < 0:
            errors.append("WEBHOOK_MAX_RETRIES must be >= 0.")

        if self.log_level not in SUPPORTED_LOG_LEVELS:
            errors.append(
                f"LOG_LEVEL must be one of {SUPPORTED_LOG_LEVELS}, "
                f"got '{self.log_level}'."
            )

        if errors:
            raise ConfigError(
                "Invalid configuration:\n  - " + "\n  - ".join(errors)
            )

    @property
    def whisper_language(self) -> str | None:
        """Return the language code for Whisper, or ``None`` for auto-detection."""
        if self.language.lower() in ("", "auto"):
            return None
        return self.language


def _parse_csv(raw: str | None) -> List[str]:
    """Parse a comma-separated env value into a clean list of strings."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
