"""Microphone capture built on :mod:`sounddevice`.

:class:`AudioRecorder` opens a single persistent 16 kHz mono input stream and
exposes incoming audio as fixed-size frames through a thread-safe queue. The
stream is opened once and reused; consumers (VAD, wake word) pull frames as
needed.

The recorder is resilient: microphone initialization failures raise a clear
:class:`MicrophoneError`, and transient stream errors are logged and surfaced so
the application loop can attempt recovery instead of crashing.
"""

from __future__ import annotations

import queue
from types import TracebackType
from typing import Iterator, Optional, Type

import numpy as np

try:  # sounddevice is optional at import time so tests can run without audio HW.
    import sounddevice as sd
except (OSError, ImportError):  # pragma: no cover - depends on host audio libs
    sd = None  # type: ignore[assignment]

from utils.logger import get_logger

logger = get_logger("stt.audio_recorder")


class MicrophoneError(RuntimeError):
    """Raised when the microphone cannot be initialized or read."""


class AudioRecorder:
    """Continuous microphone reader producing fixed-size float32 frames.

    Args:
        sample_rate: Capture sample rate in Hz.
        frame_duration: Length of each emitted frame in seconds. Silero VAD and
            openWakeWord both work well with short frames (~32 ms / 512 samples
            at 16 kHz).
        device: Optional input device index or name passed to sounddevice.
        max_queue_frames: Upper bound on buffered frames before the oldest are
            dropped, preventing unbounded memory growth if a consumer stalls.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.032,
        device: Optional[int | str] = None,
        max_queue_frames: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(sample_rate * frame_duration))
        self.device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue_frames)
        self._stream: Optional["sd.InputStream"] = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Open and start the input stream.

        Raises:
            MicrophoneError: If sounddevice is unavailable or the device fails
                to open.
        """
        if sd is None:
            raise MicrophoneError(
                "The 'sounddevice' library (and PortAudio) is not available. "
                "Install it with `pip install sounddevice` and ensure PortAudio "
                "is present on your system."
            )
        if self._stream is not None:
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
            logger.info(
                "Microphone initialized (rate=%d Hz, frame=%d samples).",
                self.sample_rate,
                self.frame_size,
            )
        except Exception as exc:  # noqa: BLE001 - normalize to MicrophoneError
            self._stream = None
            raise MicrophoneError(
                f"Failed to initialize microphone: {exc}. "
                "Check that a microphone is connected and not in use by another "
                "application."
            ) from exc

    def stop(self) -> None:
        """Stop and close the input stream, ignoring shutdown errors."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Error while closing microphone stream: %s", exc)
            finally:
                self._stream = None
                logger.debug("Microphone stream closed.")

    # -- stream callback --------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """sounddevice callback: enqueue a copy of each captured frame."""
        if status:
            # Overflows/underruns are non-fatal; log at debug to avoid spam.
            logger.debug("Audio stream status: %s", status)
        frame = np.squeeze(indata).copy().astype(np.float32)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame to keep latency bounded.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except queue.Empty:  # pragma: no cover - race window
                pass

    # -- consumption ------------------------------------------------------
    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the next audio frame, or ``None`` on timeout.

        Args:
            timeout: Seconds to wait for a frame before returning ``None``.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield frames indefinitely until the stream is stopped."""
        while self._stream is not None:
            frame = self.read_frame(timeout=timeout)
            if frame is not None:
                yield frame

    def flush(self) -> None:
        """Discard any buffered frames (e.g. before starting a new capture)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        """Whether the input stream is currently open."""
        return self._stream is not None

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "AudioRecorder":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.stop()
