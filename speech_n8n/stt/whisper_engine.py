"""Faster-Whisper transcription engine.

:class:`WhisperEngine` wraps a single Faster-Whisper ``WhisperModel`` instance.
The model is loaded exactly once (lazily on first use, or eagerly via
:meth:`load`) and reused for every transcription, satisfying the requirement to
never reload the model per request.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("stt.whisper_engine")


@dataclass
class TranscriptionResult:
    """Structured result of a transcription."""

    transcribed_text: str
    detected_language: str
    confidence_score: float
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain dictionary."""
        return asdict(self)


class WhisperEngine:
    """Local speech-to-text using Faster-Whisper.

    Args:
        model_size: One of tiny | base | small | medium | large-v3.
        device: ``cpu``, ``cuda`` or ``auto``.
        compute_type: e.g. ``int8`` (CPU) or ``float16`` (GPU).
        beam_size: Decoding beam size.
        language: Forced language code, or ``None`` for auto-detection.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        language: Optional[str] = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model: Optional[Any] = None

    def load(self) -> None:
        """Load the Whisper model once. Subsequent calls are no-ops."""
        if self._model is not None:
            return

        # Imported lazily so the rest of the app (and tests) can run without the
        # heavy faster-whisper dependency installed.
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        start = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(
            "Whisper model loaded in %.2fs.", time.perf_counter() - start
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been loaded."""
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe float32 mono audio sampled at 16 kHz.

        Args:
            audio: 1-D float32 numpy array in [-1.0, 1.0].

        Returns:
            A :class:`TranscriptionResult`.
        """
        if self._model is None:
            # Defensive: ensure we never silently reload per call, but also never
            # crash if `load()` was skipped.
            self.load()

        assert self._model is not None  # for type checkers

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        start = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,  # We already gate audio with Silero VAD upstream.
        )

        texts: list[str] = []
        logprobs: list[float] = []
        for segment in segments:  # generator must be consumed to run inference
            texts.append(segment.text)
            # avg_logprob is per-segment; collect to derive a confidence score.
            if getattr(segment, "avg_logprob", None) is not None:
                logprobs.append(segment.avg_logprob)

        processing_time = time.perf_counter() - start
        text = "".join(texts).strip()
        confidence = _logprob_to_confidence(logprobs)

        result = TranscriptionResult(
            transcribed_text=text,
            detected_language=getattr(info, "language", self.language or "unknown"),
            confidence_score=confidence,
            processing_time=processing_time,
        )
        logger.info(
            "Transcription complete in %.2fs (lang=%s, conf=%.2f): '%s'",
            processing_time,
            result.detected_language,
            result.confidence_score,
            text,
        )
        return result


def _logprob_to_confidence(logprobs: list[float]) -> float:
    """Map average segment log-probabilities to a 0..1 confidence score.

    Whisper does not emit a calibrated confidence; we approximate one by taking
    ``exp(mean(avg_logprob))`` which yields the geometric-mean token
    probability. Returns ``0.0`` when no segments were produced.
    """
    if not logprobs:
        return 0.0
    mean_logprob = sum(logprobs) / len(logprobs)
    return round(float(math.exp(mean_logprob)), 4)
