"""Tests for audio utility helpers."""

import wave

import numpy as np
import pytest

from utils.audio_utils import (
    duration_seconds,
    float_to_int16,
    int16_to_float,
    save_wav,
)


def test_float_to_int16_clips():
    audio = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = float_to_int16(audio)
    assert pcm.dtype == np.int16
    assert pcm[1] == 32767
    assert pcm[3] == 32767  # clipped
    assert pcm[4] == -32767  # clipped


def test_int16_float_roundtrip():
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    restored = int16_to_float(float_to_int16(audio))
    assert np.allclose(audio, restored, atol=1e-3)


def test_duration_seconds():
    assert duration_seconds(16000, 16000) == 1.0
    assert duration_seconds(8000, 16000) == 0.5


def test_duration_seconds_invalid_rate():
    with pytest.raises(ValueError):
        duration_seconds(100, 0)


def test_save_wav_writes_file(tmp_path):
    audio = np.zeros(16000, dtype=np.float32)
    path = save_wav(audio, 16000, str(tmp_path))
    assert path is not None
    with wave.open(path, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2


def test_save_wav_disabled_returns_none():
    assert save_wav(np.zeros(10, dtype=np.float32), 16000, "") is None
