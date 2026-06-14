"""Tests for the VAD utterance collector using a scripted fake VAD.

The collector logic (speech start, trailing-silence end, min-duration gating,
max-duration cap, pre-roll) is tested without loading Silero by injecting a fake
VAD whose ``is_speech`` is driven by a flag.
"""

import numpy as np

from vad.vad_engine import UtteranceCollector


class FakeVad:
    """VAD whose speech decision follows the mean of the frame's samples."""

    window_size = 512

    def is_speech(self, frame: np.ndarray) -> bool:
        # Frames with positive mean are "speech"; zeros are "silence".
        return float(np.mean(frame)) > 0.0


def _frames(pattern, window=512):
    """Build a list of frames; 1 -> speech frame, 0 -> silence frame."""
    out = []
    for token in pattern:
        if token == 1:
            out.append(np.ones(window, dtype=np.float32))
        else:
            out.append(np.zeros(window, dtype=np.float32))
    return out


def _source_from(frames):
    """Return a frame source callable that yields frames then None forever."""
    iterator = iter(frames)

    def _next():
        return next(iterator, None)

    return _next


def test_collects_speech_then_ends_on_silence():
    sample_rate = 16000  # window=512 -> 32 ms per frame
    collector = UtteranceCollector(
        vad=FakeVad(),
        sample_rate=sample_rate,
        silence_timeout=0.064,  # 2 silence windows end the utterance
        min_speech_duration=0.03,  # ~1 window of speech qualifies
        max_recording_seconds=5.0,
        preroll_seconds=0.0,
    )
    # 5 speech frames, then 3 silence frames to trigger end.
    frames = _frames([1, 1, 1, 1, 1, 0, 0, 0])
    result = collector.collect(_source_from(frames))

    assert result is not None
    assert result.reason == "silence"
    assert result.duration > 0


def test_short_utterance_discarded():
    collector = UtteranceCollector(
        vad=FakeVad(),
        sample_rate=16000,
        silence_timeout=0.064,
        min_speech_duration=1.0,  # require 1s of speech (impossible here)
        max_recording_seconds=5.0,
        preroll_seconds=0.0,
    )
    frames = _frames([1, 0, 0, 0])
    result = collector.collect(_source_from(frames))
    assert result is None


def test_max_duration_cap():
    sample_rate = 16000
    collector = UtteranceCollector(
        vad=FakeVad(),
        sample_rate=sample_rate,
        silence_timeout=10.0,  # never reached
        min_speech_duration=0.0,
        max_recording_seconds=0.096,  # 3 windows
        preroll_seconds=0.0,
    )
    frames = _frames([1] * 10)
    result = collector.collect(_source_from(frames))
    assert result is not None
    assert result.reason == "max_duration"


def test_preroll_included():
    sample_rate = 16000
    collector = UtteranceCollector(
        vad=FakeVad(),
        sample_rate=sample_rate,
        silence_timeout=0.064,
        min_speech_duration=0.03,
        max_recording_seconds=5.0,
        preroll_seconds=0.064,  # keep ~2 windows of pre-roll
    )
    # Two silence frames (pre-roll), then speech, then trailing silence.
    frames = _frames([0, 0, 1, 1, 0, 0, 0])
    result = collector.collect(_source_from(frames))
    assert result is not None
    # Audio should include the pre-roll silence captured before speech began.
    assert result.audio.size >= 4 * collector.vad.window_size
