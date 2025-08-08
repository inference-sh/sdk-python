import io
import json
import os
import types
from contextlib import contextmanager
from typing import Iterator

import pytest

from inferencesh import Inference


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", lines=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {"success": True, "data": {}}
        self.text = text
        self._lines = lines or []

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP error {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def close(self):
        pass


@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    calls = []

    def fake_request(method, url, params=None, data=None, headers=None, stream=False, timeout=None):
        calls.append({
            "method": method,
            "url": url,
            "params": params,
            "data": data,
            "headers": headers,
            "stream": stream,
            "timeout": timeout,
        })

        # Create task
        if url.endswith("/run") and method.upper() == "POST":
            body = json.loads(data)
            return DummyResponse(json_data={
                "success": True,
                "data": {
                    "id": "task_123",
                    "status": 1,
                    "input": body.get("input"),
                },
            })

        # SSE stream
        if url.endswith("/tasks/task_123/stream") and stream:
            # Minimal SSE: send a completed event
            event_payload = json.dumps({
                "status": 9,  # COMPLETED
                "output": {"ok": True},
                "logs": ["done"],
            })
            lines = [
                f"data: {event_payload}",
                "",  # dispatch
            ]
            return DummyResponse(status_code=200, lines=lines)

        # Cancel
        if url.endswith("/tasks/task_123/cancel") and method.upper() == "POST":
            return DummyResponse(json_data={"success": True, "data": None})

        # Files create
        if url.endswith("/files") and method.upper() == "POST":
            upload_url = "https://upload.example.com/file"
            return DummyResponse(json_data={
                "success": True,
                "data": [
                    {
                        "id": "file_1",
                        "uri": "https://cloud.inference.sh/u/user/file_1.png",
                        "upload_url": upload_url,
                    }
                ],
            })

        return DummyResponse()

    class FakeRequestsModule:
        def __init__(self):
            self.put_calls = []

        def request(self, *args, **kwargs):
            return fake_request(*args, **kwargs)

        def put(self, url, data=None, headers=None):
            self.put_calls.append({"url": url, "size": len(data or b"")})
            return DummyResponse(status_code=200)

    fake_requests = FakeRequestsModule()

    def require_requests():
        return fake_requests

    # Patch internal loader
    from inferencesh import client as client_mod
    monkeypatch.setattr(client_mod, "_require_requests", require_requests)

    yield fake_requests


def test_run_and_run_sync_mocked(tmp_path):
    client = Inference(api_key="test")

    # run() should return a task id
    task = client.run({
        "app": "some/app",
        "input": {"text": "hello"},
        "worker_selection_mode": "private",
    })
    assert task["id"] == "task_123"

    # run_sync should consume SSE and return final result merged
    result = client.run_sync({
        "app": "some/app",
        "input": {"text": "hello"},
        "worker_selection_mode": "private",
    })
    assert result["id"] == "task_123"
    assert result["output"] == {"ok": True}
    assert result["logs"] == ["done"]


def test_upload_and_recursive_input(monkeypatch, tmp_path, patch_requests):
    # Create a small file
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"PNGDATA")

    client = Inference(api_key="test")

    # Input contains a local path - should be uploaded and replaced by uri before /run
    task = client.run({
        "app": "some/app",
        "input": {"image": str(file_path)},
        "worker_selection_mode": "private",
    })

    # The mocked /run echoes input; ensure it is not the raw path anymore (upload replaced it)
    assert task["input"]["image"] != str(file_path)
    assert task["input"]["image"].startswith("https://cloud.inference.sh/")


