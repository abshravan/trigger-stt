"""Tests for configuration loading and validation."""

import pytest

from config import Config, ConfigError


def _base_env(monkeypatch, **overrides):
    """Set a minimal valid environment, applying overrides."""
    env = {
        "N8N_WEBHOOK_URL": "http://localhost:5678/webhook/test",
        "WHISPER_MODEL": "base",
        "DEVICE": "cpu",
        "COMPUTE_TYPE": "int8",
        "LANGUAGE": "en",
        "WAKEWORD_ENABLED": "false",
        "SILENCE_TIMEOUT": "1.5",
        "MAX_RECORDING_SECONDS": "20",
        "LOG_LEVEL": "INFO",
    }
    env.update(overrides)
    for key in list(env.keys()):
        monkeypatch.setenv(key, env[key])
    return env


def test_from_env_loads_valid_config(monkeypatch):
    _base_env(monkeypatch)
    config = Config.from_env(env_file=None)
    assert config.n8n_webhook_url == "http://localhost:5678/webhook/test"
    assert config.whisper_model == "base"
    assert config.device == "cpu"
    assert config.wakeword_enabled is False
    assert config.silence_timeout == 1.5


def test_missing_webhook_url_fails(monkeypatch):
    _base_env(monkeypatch, N8N_WEBHOOK_URL="")
    with pytest.raises(ConfigError) as exc:
        Config.from_env(env_file=None)
    assert "N8N_WEBHOOK_URL" in str(exc.value)


def test_invalid_webhook_scheme_fails(monkeypatch):
    _base_env(monkeypatch, N8N_WEBHOOK_URL="ftp://example.com/hook")
    with pytest.raises(ConfigError):
        Config.from_env(env_file=None)


def test_invalid_model_fails(monkeypatch):
    _base_env(monkeypatch, WHISPER_MODEL="gigantic")
    with pytest.raises(ConfigError) as exc:
        Config.from_env(env_file=None)
    assert "WHISPER_MODEL" in str(exc.value)


def test_invalid_compute_type_fails(monkeypatch):
    _base_env(monkeypatch, COMPUTE_TYPE="float128")
    with pytest.raises(ConfigError):
        Config.from_env(env_file=None)


def test_vad_threshold_out_of_range_fails(monkeypatch):
    _base_env(monkeypatch, VAD_THRESHOLD="1.5")
    with pytest.raises(ConfigError):
        Config.from_env(env_file=None)


def test_boolean_parsing(monkeypatch):
    _base_env(monkeypatch, WAKEWORD_ENABLED="yes")
    config = Config.from_env(env_file=None)
    assert config.wakeword_enabled is True


def test_wakeword_models_csv_parsing(monkeypatch):
    _base_env(monkeypatch, WAKEWORD_MODELS="hey_jarvis, computer ,alexa")
    config = Config.from_env(env_file=None)
    assert config.wakeword_models == ["hey_jarvis", "computer", "alexa"]


def test_language_auto_returns_none(monkeypatch):
    _base_env(monkeypatch, LANGUAGE="auto")
    config = Config.from_env(env_file=None)
    assert config.whisper_language is None


def test_language_explicit_returned(monkeypatch):
    _base_env(monkeypatch, LANGUAGE="es")
    config = Config.from_env(env_file=None)
    assert config.whisper_language == "es"


def test_wakeword_enabled_requires_models():
    config = Config(
        n8n_webhook_url="http://x/hook",
        wakeword_enabled=True,
        wakeword_models=[],
    )
    with pytest.raises(ConfigError):
        config.validate()
