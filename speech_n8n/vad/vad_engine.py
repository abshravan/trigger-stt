"""Voice Activity Detection using Silero VAD.

:class:`VadEngine` consumes short audio frames and emits a per-frame speech
probability. :class:`UtteranceCollector` builds on top of it to turn a stream of
frames into a single utterance: it waits for speech to start, keeps a short
pre-roll so the beginning is not clipped, and ends the utterance after a
configurable trailing-silence duration.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("vad.vad_engine")

# Silero VAD expects exactly 512 samples per call at 16 kHz (256 at 8 kHz).
_SILERO_FRAME_16K = 512
_SILERO_FRAME_8K = 256


class VadEngine:
    """Thin wrapper around the Silero VAD model.

    Args:
        sample_rate: Audio sample rate (8000 or 16000 Hz).
        threshold: Speech probability above which a frame counts as speech.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz audio.")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._window = _SILERO_FRAME_16K if sample_rate == 16000 else _SILERO_FRAME_8K
        self._model: Optional[object] = None

    def load(self) -> None:
        """Load the Silero VAD model once."""
        if self._model is not None:
            return
        from silero_vad import load_silero_vad

        logger.info("Loading Silero VAD model...")
        self._model = load_silero_vad()
        logger.info("Silero VAD model loaded.")

    @property
    def window_size(self) -> int:
        """Number of samples Silero expects per inference call."""
        return self._window

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return the speech probability for a single ``window_size`` frame.

        Args:
            frame: Float32 audio of exactly :attr:`window_size` samples.
        """
        if self._model is None:
            self.load()
        import torch

        if len(frame) != self._window:
            frame = _fit_window(frame, self._window)

        tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()  # type: ignore[operator]
        return float(prob)

    def is_speech(self, frame: np.ndarray) -> bool:
        """Whether a frame's speech probability exceeds the threshold."""
        return self.speech_probability(frame) >= self.threshold


def _fit_window(frame: np.ndarray, window: int) -> np.ndarray:
    """Pad or trim a frame to exactly ``window`` samples."""
    if len(frame) > window:
        return frame[:window]
    return np.pad(frame, (0, window - len(frame)))


class _State(Enum):
    WAITING = auto()
    SPEAKING = auto()


@dataclass
class UtteranceResult:
    """Outcome of collecting a single utterance."""

    audio: np.ndarray
    duration: float
    reason: str  # "silence" | "max_duration"


class UtteranceCollector:
    """Assemble a single spoken utterance from a frame stream using VAD.

    The collector keeps a rolling pre-roll buffer so that audio captured just
    before speech is detected is included, preventing the first word from being
    truncated.

    Args:
        vad: A loaded :class:`VadEngine`.
        sample_rate: Audio sample rate in Hz.
        silence_timeout: Trailing silence (seconds) that ends an utterance.
        min_speech_duration: Minimum speech (seconds) for a valid utterance.
        max_recording_seconds: Hard cap on utterance length.
        preroll_seconds: Amount of pre-speech audio to retain.
    """

    def __init__(
        self,
        vad: VadEngine,
        sample_rate: int = 16000,
        silence_timeout: float = 1.5,
        min_speech_duration: float = 0.25,
        max_recording_seconds: float = 20.0,
        preroll_seconds: float = 0.3,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.max_recording_seconds = max_recording_seconds
        preroll_frames = max(1, int(preroll_seconds * sample_rate / vad.window_size))
        self._preroll: Deque[np.ndarray] = collections.deque(maxlen=preroll_frames)

    def collect(self, frame_source: "FrameSource") -> Optional[UtteranceResult]:
        """Block until one utterance is captured or the source is exhausted.

        Args:
            frame_source: Callable returning the next float32 frame, or ``None``
                when no audio is currently available.

        Returns:
            An :class:`UtteranceResult`, or ``None`` if the stream ended before a
            valid utterance was captured.
        """
        state = _State.WAITING
        collected: List[np.ndarray] = []
        speech_samples = 0
        silence_samples = 0
        total_samples = 0
        silence_limit = int(self.silence_timeout * self.sample_rate)
        max_samples = int(self.max_recording_seconds * self.sample_rate)
        window = self.vad.window_size

        while True:
            frame = frame_source()
            if frame is None:
                if state == _State.SPEAKING:
                    # Stream paused mid-utterance; treat as silence accumulation.
                    silence_samples += window
                    if silence_samples >= silence_limit:
                        return self._finalize(collected, speech_samples, "silence")
                continue

            # Normalize to the VAD window size.
            for sub in _split_into_windows(frame, window):
                speech = self.vad.is_speech(sub)

                if state == _State.WAITING:
                    self._preroll.append(sub)
                    if speech:
                        logger.info("Speech detected.")
                        collected = list(self._preroll)
                        collected.append(sub)
                        speech_samples = window
                        silence_samples = 0
                        total_samples = len(collected) * window
                        state = _State.SPEAKING
                else:  # SPEAKING
                    collected.append(sub)
                    total_samples += window
                    if speech:
                        speech_samples += window
                        silence_samples = 0
                    else:
                        silence_samples += window

                    if silence_samples >= silence_limit:
                        return self._finalize(
                            collected, speech_samples, "silence"
                        )
                    if total_samples >= max_samples:
                        logger.warning(
                            "Max recording length reached (%.1fs).",
                            self.max_recording_seconds,
                        )
                        return self._finalize(
                            collected, speech_samples, "max_duration"
                        )

    def _finalize(
        self, collected: List[np.ndarray], speech_samples: int, reason: str
    ) -> Optional[UtteranceResult]:
        """Build a result, discarding utterances shorter than the minimum."""
        speech_duration = speech_samples / float(self.sample_rate)
        if speech_duration < self.min_speech_duration:
            logger.debug(
                "Discarded short utterance (%.2fs < %.2fs).",
                speech_duration,
                self.min_speech_duration,
            )
            self._preroll.clear()
            return None

        audio = np.concatenate(collected) if collected else np.zeros(0, np.float32)
        self._preroll.clear()
        return UtteranceResult(
            audio=audio,
            duration=len(audio) / float(self.sample_rate),
            reason=reason,
        )


def _split_into_windows(frame: np.ndarray, window: int) -> List[np.ndarray]:
    """Split an arbitrary-length frame into ``window``-sized chunks.

    The final partial chunk is zero-padded so Silero always receives a full
    window.
    """
    if len(frame) == window:
        return [frame]
    chunks: List[np.ndarray] = []
    for start in range(0, len(frame), window):
        chunk = frame[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        chunks.append(chunk)
    return chunks


# Type alias documenting the expected frame source signature.
from typing import Callable  # noqa: E402  (placed here to keep imports grouped)

FrameSource = Callable[[], Optional[np.ndarray]]
