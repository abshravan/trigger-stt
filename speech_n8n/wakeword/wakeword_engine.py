"""Optional wake word detection using openWakeWord.

The application defines a small :class:`WakeWordDetector` protocol so the main
loop can treat "wake word enabled" and "wake word disabled" uniformly. When
disabled, :class:`NullWakeWordDetector` is injected and reports the wake word as
always satisfied, so recording starts immediately on speech.

This keeps wake word support fully optional and swappable without touching the
core pipeline (Feature 5: enable/disable without major changes).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

from utils.logger import get_logger

logger = get_logger("wakeword.wakeword_engine")


@runtime_checkable
class WakeWordDetector(Protocol):
    """Interface implemented by all wake word detectors."""

    enabled: bool

    def reset(self) -> None:
        """Clear any internal detection state."""

    def process(self, frame: np.ndarray) -> Optional[str]:
        """Process one audio frame.

        Returns:
            The name of the detected wake word, or ``None`` if none fired.
        """


class NullWakeWordDetector:
    """No-op detector used when wake word support is disabled."""

    enabled = False

    def reset(self) -> None:  # noqa: D102 - trivial
        return None

    def process(self, frame: np.ndarray) -> Optional[str]:  # noqa: D102
        return None


class OpenWakeWordDetector:
    """Wake word detector backed by openWakeWord.

    Args:
        models: openWakeWord model names (e.g. ``["hey_jarvis"]``).
        threshold: Score above which a detection fires.
        sample_rate: Audio sample rate; openWakeWord expects 16 kHz.
    """

    enabled = True

    def __init__(
        self,
        models: List[str],
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.models = models
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model: Optional[object] = None

    def load(self) -> None:
        """Load the openWakeWord model(s) once."""
        if self._model is not None:
            return
        from openwakeword.model import Model

        logger.info("Loading wake word models: %s", ", ".join(self.models))
        self._model = Model(wakeword_models=self.models)
        logger.info("Wake word models loaded.")

    def reset(self) -> None:
        """Reset openWakeWord's internal streaming buffers."""
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()  # type: ignore[attr-defined]

    def process(self, frame: np.ndarray) -> Optional[str]:
        """Feed a frame to openWakeWord and report any detection.

        Args:
            frame: Float32 audio in [-1.0, 1.0]; converted to int16 internally.
        """
        if self._model is None:
            self.load()

        # openWakeWord expects 16-bit PCM samples.
        pcm = np.clip(frame, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        predictions = self._model.predict(pcm)  # type: ignore[attr-defined]

        for name, score in predictions.items():
            if score >= self.threshold:
                logger.info("Wake word detected: '%s' (score=%.2f).", name, score)
                self.reset()
                return name
        return None


def create_detector(
    enabled: bool,
    models: List[str],
    threshold: float,
    sample_rate: int,
) -> WakeWordDetector:
    """Factory returning the appropriate detector based on configuration.

    Args:
        enabled: Whether wake word detection is enabled.
        models: openWakeWord model names.
        threshold: Detection threshold.
        sample_rate: Audio sample rate in Hz.

    Returns:
        An :class:`OpenWakeWordDetector` when enabled, else a
        :class:`NullWakeWordDetector`.
    """
    if not enabled:
        return NullWakeWordDetector()
    return OpenWakeWordDetector(
        models=models, threshold=threshold, sample_rate=sample_rate
    )
