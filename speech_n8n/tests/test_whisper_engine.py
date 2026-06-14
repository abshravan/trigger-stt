"""Tests for the Whisper engine interface using a fake backend.

These tests avoid loading the real Faster-Whisper model by injecting a fake
``WhisperModel`` into the engine, verifying load-once behaviour and the
structured result contract.
"""

import math
import types

import numpy as np

from stt.whisper_engine import (
    TranscriptionResult,
    WhisperEngine,
    _logprob_to_confidence,
)


class _Segment:
    def __init__(self, text, avg_logprob):
        self.text = text
        self.avg_logprob = avg_logprob


class _Info:
    language = "en"


class FakeWhisperModel:
    """Counts loads and transcribe calls; returns canned segments."""

    instances = 0

    def __init__(self, *args, **kwargs):
        FakeWhisperModel.instances += 1
        self.transcribe_calls = 0

    def transcribe(self, audio, beam_size=5, language=None, vad_filter=False):
        self.transcribe_calls += 1
        segments = [_Segment(" hello", -0.2), _Segment(" world", -0.4)]
        return iter(segments), _Info()


def _install_fake(monkeypatch):
    """Patch faster_whisper.WhisperModel with the fake."""
    FakeWhisperModel.instances = 0
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", fake_module)


def test_model_loaded_once(monkeypatch):
    _install_fake(monkeypatch)
    engine = WhisperEngine(model_size="base")
    engine.load()
    engine.load()  # second call must be a no-op
    assert engine.is_loaded
    assert FakeWhisperModel.instances == 1


def test_transcribe_returns_structured_result(monkeypatch):
    _install_fake(monkeypatch)
    engine = WhisperEngine(model_size="base")
    engine.load()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio)

    assert isinstance(result, TranscriptionResult)
    assert result.transcribed_text == "hello world"
    assert result.detected_language == "en"
    assert 0.0 < result.confidence_score <= 1.0
    assert result.processing_time >= 0.0

    keys = set(result.to_dict().keys())
    assert keys == {
        "transcribed_text",
        "detected_language",
        "confidence_score",
        "processing_time",
    }


def test_transcribe_does_not_reload(monkeypatch):
    _install_fake(monkeypatch)
    engine = WhisperEngine()
    engine.load()
    audio = np.zeros(8000, dtype=np.float32)
    engine.transcribe(audio)
    engine.transcribe(audio)
    assert FakeWhisperModel.instances == 1


def test_logprob_to_confidence_empty():
    assert _logprob_to_confidence([]) == 0.0


def test_logprob_to_confidence_value():
    # exp(mean([-0.2, -0.4])) = exp(-0.3)
    conf = _logprob_to_confidence([-0.2, -0.4])
    assert math.isclose(conf, math.exp(-0.3), rel_tol=1e-3)
