"""Tests for the n8n webhook client: payload, retries, error handling."""

from typing import List

import pytest

from webhook.n8n_client import (
    N8nClient,
    SOURCE_NAME,
    WebhookResponse,
    build_payload,
)


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, status_code: int, json_body=None, text: str = ""):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self.headers = {
            "Content-Type": "application/json" if json_body is not None else "text/plain"
        }

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Session that returns scripted responses or raises scripted errors."""

    def __init__(self, outcomes: List[object]):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.last_json = None

    def post(self, url, json=None, timeout=None):  # noqa: A002 - mirror requests API
        self.calls += 1
        self.last_json = json
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


# -- payload ----------------------------------------------------------------
def test_build_payload_fields():
    payload = build_payload(
        text="Turn on the lights",
        language="en",
        confidence=0.964321,
        timestamp="2026-06-14T00:00:00+00:00",
    )
    assert payload["text"] == "Turn on the lights"
    assert payload["language"] == "en"
    assert payload["confidence"] == 0.9643  # rounded to 4 dp
    assert payload["source"] == SOURCE_NAME
    assert payload["timestamp"] == "2026-06-14T00:00:00+00:00"


def test_build_payload_generates_timestamp():
    payload = build_payload(text="hi", language="en", confidence=0.5)
    assert payload["timestamp"]  # non-empty ISO timestamp


# -- success ----------------------------------------------------------------
def test_send_success_returns_parsed_body():
    session = FakeSession([FakeResponse(200, json_body={"reply": "ok"})])
    client = N8nClient("http://x/hook", session=session)
    resp = client.send({"text": "hi"})
    assert isinstance(resp, WebhookResponse)
    assert resp.ok is True
    assert resp.status_code == 200
    assert resp.body == {"reply": "ok"}
    assert session.calls == 1


# -- retries ----------------------------------------------------------------
def test_send_retries_on_server_error(monkeypatch):
    monkeypatch.setattr("webhook.n8n_client.time.sleep", lambda *_: None)
    session = FakeSession(
        [FakeResponse(500), FakeResponse(503), FakeResponse(200, json_body={"ok": 1})]
    )
    client = N8nClient("http://x/hook", max_retries=3, session=session)
    resp = client.send({"text": "hi"})
    assert resp.ok is True
    assert session.calls == 3


def test_send_retries_on_network_error(monkeypatch):
    import requests

    monkeypatch.setattr("webhook.n8n_client.time.sleep", lambda *_: None)
    session = FakeSession(
        [requests.ConnectionError("boom"), FakeResponse(200, json_body={"ok": 1})]
    )
    client = N8nClient("http://x/hook", max_retries=2, session=session)
    resp = client.send({"text": "hi"})
    assert resp.ok is True
    assert session.calls == 2


def test_send_exhausts_retries_and_fails(monkeypatch):
    monkeypatch.setattr("webhook.n8n_client.time.sleep", lambda *_: None)
    session = FakeSession([FakeResponse(500), FakeResponse(500)])
    client = N8nClient("http://x/hook", max_retries=1, session=session)
    resp = client.send({"text": "hi"})
    assert resp.ok is False
    assert resp.status_code == 0
    assert session.calls == 2
    assert "500" in (resp.error or "")


def test_send_does_not_retry_client_error(monkeypatch):
    monkeypatch.setattr("webhook.n8n_client.time.sleep", lambda *_: None)
    session = FakeSession([FakeResponse(404, text="not found")])
    client = N8nClient("http://x/hook", max_retries=3, session=session)
    resp = client.send({"text": "hi"})
    assert resp.ok is False
    assert resp.status_code == 404
    assert session.calls == 1  # 4xx is not retried


def test_backoff_delays_are_exponential(monkeypatch):
    delays = []
    monkeypatch.setattr(
        "webhook.n8n_client.time.sleep", lambda d: delays.append(d)
    )
    session = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    client = N8nClient(
        "http://x/hook", max_retries=2, backoff_factor=0.5, session=session
    )
    client.send({"text": "hi"})
    # delay before retry 1 = 0.5 * 2**0, before retry 2 = 0.5 * 2**1
    assert delays == [0.5, 1.0]
