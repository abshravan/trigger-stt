"""HTTP client for posting transcriptions to an n8n webhook.

:class:`N8nClient` performs a POST with a JSON payload, retries transient
failures with exponential backoff, and returns the parsed response. It logs
every request, response and retry attempt.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from utils.logger import get_logger

logger = get_logger("webhook.n8n_client")

SOURCE_NAME = "desktop_voice_agent"


@dataclass
class WebhookResponse:
    """Parsed n8n webhook response."""

    ok: bool
    status_code: int
    body: Any  # parsed JSON when possible, otherwise raw text
    error: Optional[str] = None


def build_payload(
    text: str,
    language: str,
    confidence: float,
    timestamp: Optional[str] = None,
    source: str = SOURCE_NAME,
) -> Dict[str, Any]:
    """Build the JSON payload sent to n8n.

    Args:
        text: Transcribed text.
        language: Detected/forced language code.
        confidence: Confidence score in 0..1.
        timestamp: ISO-8601 timestamp; generated (UTC) when omitted.
        source: Identifier for the sending application.

    Returns:
        A JSON-serializable dictionary.
    """
    return {
        "text": text,
        "timestamp": timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "language": language,
        "confidence": round(float(confidence), 4),
        "source": source,
    }


class N8nClient:
    """POST transcriptions to an n8n webhook with retries and backoff.

    Args:
        webhook_url: Destination webhook URL.
        timeout: Per-request timeout in seconds.
        max_retries: Number of additional attempts after the first failure.
        backoff_factor: Base for exponential backoff; the delay before retry
            ``n`` (1-indexed) is ``backoff_factor * 2 ** (n - 1)`` seconds.
        session: Optional pre-configured :class:`requests.Session` (dependency
            injection for testing / connection reuse).
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = session or requests.Session()

    def send(self, payload: Dict[str, Any]) -> WebhookResponse:
        """Send a payload, retrying transient failures with backoff.

        Args:
            payload: JSON-serializable request body.

        Returns:
            A :class:`WebhookResponse`. ``ok`` is ``False`` if every attempt
            failed; the method does not raise on network/HTTP errors.
        """
        attempts = self.max_retries + 1
        last_error: Optional[str] = None

        for attempt in range(1, attempts + 1):
            logger.info(
                "Sending to n8n (attempt %d/%d): %s",
                attempt,
                attempts,
                self.webhook_url,
            )
            try:
                response = self._session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"Request error: {exc}"
                logger.warning("Webhook attempt %d failed: %s", attempt, exc)
            else:
                if response.status_code < 500:
                    # 2xx/3xx success, or 4xx client errors that retrying won't fix.
                    parsed = _parse_body(response)
                    ok = response.ok
                    logger.info(
                        "n8n responded with HTTP %d (ok=%s).",
                        response.status_code,
                        ok,
                    )
                    return WebhookResponse(
                        ok=ok,
                        status_code=response.status_code,
                        body=parsed,
                        error=None if ok else f"HTTP {response.status_code}",
                    )
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    "Webhook attempt %d returned server error %d.",
                    attempt,
                    response.status_code,
                )

            if attempt < attempts:
                delay = self.backoff_factor * (2 ** (attempt - 1))
                logger.info("Retrying in %.2fs...", delay)
                time.sleep(delay)

        logger.error("All %d webhook attempts failed: %s", attempts, last_error)
        return WebhookResponse(
            ok=False, status_code=0, body=None, error=last_error
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


def _parse_body(response: requests.Response) -> Any:
    """Return the JSON body when possible, else the raw text."""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text
