"""Small, dependency-light audio helper functions.

These helpers are intentionally free of heavy imports (no torch / sounddevice)
so they can be unit-tested quickly and reused across modules.
"""

from __future__ import annotations

import datetime as _dt
import os
import wave
from typing import Optional

import numpy as np


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio in [-1.0, 1.0] to int16 PCM.

    Args:
        audio: Float audio samples.

    Returns:
        ``int16`` numpy array suitable for writing to a WAV file.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM audio to float32 in [-1.0, 1.0]."""
    return (audio.astype(np.float32)) / 32768.0


def duration_seconds(num_samples: int, sample_rate: int) -> float:
    """Return the duration in seconds for a number of samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return num_samples / float(sample_rate)


def save_wav(
    audio: np.ndarray,
    sample_rate: int,
    directory: str,
    prefix: str = "utterance",
) -> Optional[str]:
    """Save float audio to a timestamped 16-bit mono WAV file.

    Args:
        audio: Float32 mono audio samples in [-1.0, 1.0].
        sample_rate: Sample rate in Hz.
        directory: Target directory; created if it does not exist.
        prefix: Filename prefix.

    Returns:
        The path to the written file, or ``None`` if ``directory`` is empty.
    """
    if not directory:
        return None

    os.makedirs(directory, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(directory, f"{prefix}_{timestamp}.wav")

    pcm = float_to_int16(audio)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return path
