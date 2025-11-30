import json
from unittest.mock import MagicMock

import pytest

from inferencesh import Inference, AsyncInference, TaskStatus


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", lines=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {"success": True, "data": {}}
        self.text = text
        self._lines = lines or []
        self.raw = None  # For SSE raw mode

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP error {self.status_code}")

    def iter_lines(self, decode_unicode=False, chunk_size=None):
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

        # Get task
        if "/tasks/task_123" in url and method.upper() == "GET" and not url.endswith("/stream"):
            return DummyResponse(json_data={
                "success": True,
                "data": {
                    "id": "task_123",
                    "status": 7,  # RUNNING
                    "input": {"text": "hello"},
                },
            })

        # SSE stream
        if url.endswith("/tasks/task_123/stream") and stream:
            # Minimal SSE: send a completed event
            event_payload = json.dumps({
                "id": "task_123",
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


def test_run_wait_false(tmp_path):
    """Test run() with wait=False returns task info immediately."""
    client = Inference(api_key="test")

    # run(wait=False) should return task info without waiting
    task = client.run({
        "app": "some/app",
        "input": {"text": "hello"},
        "worker_selection_mode": "private",
    }, wait=False)
    
    assert task["id"] == "task_123"
    assert task["status"] == 1  # RECEIVED


def test_run_wait_true(tmp_path):
    """Test run() with wait=True (default) waits for completion."""
    client = Inference(api_key="test")

    # run() with default wait=True should wait for completion
    result = client.run({
        "app": "some/app",
        "input": {"text": "hello"},
        "worker_selection_mode": "private",
    })
    
    assert result["id"] == "task_123"
    assert result["output"] == {"ok": True}
    assert result["logs"] == ["done"]
    assert result["status"] == TaskStatus.COMPLETED


def test_run_stream(tmp_path):
    """Test run() with stream=True returns iterator of updates."""
    client = Inference(api_key="test")

    updates = []
    for update in client.run({
        "app": "some/app",
        "input": {"text": "hello"},
        "worker_selection_mode": "private",
    }, stream=True):
        updates.append(update)
        if update.get("status") == TaskStatus.COMPLETED:
            break

    assert len(updates) >= 1
    final = updates[-1]
    assert final["id"] == "task_123"
    assert final["output"] == {"ok": True}
    assert final["status"] == TaskStatus.COMPLETED


def test_get_task(tmp_path):
    """Test get_task() returns current task state."""
    client = Inference(api_key="test")

    task = client.get_task("task_123")
    
    assert task["id"] == "task_123"
    assert task["status"] == 7  # RUNNING


def test_cancel(tmp_path):
    """Test cancel() cancels a task."""
    client = Inference(api_key="test")

    # Should not raise
    client.cancel("task_123")


def test_stream_task(tmp_path):
    """Test stream_task() returns TaskStream for existing task."""
    client = Inference(api_key="test")

    with client.stream_task("task_123") as stream:
        updates = []
        for update in stream:
            updates.append(update)
            if update.get("status") == TaskStatus.COMPLETED:
                break

    assert len(updates) >= 1
    assert stream.result is not None
    assert stream.result["output"] == {"ok": True}


def test_upload_and_recursive_input(monkeypatch, tmp_path, patch_requests):
    """Test that local file paths in input are uploaded and replaced with URIs."""
    # Create a small file
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"PNGDATA")

    client = Inference(api_key="test")

    # Input contains a local path - should be uploaded and replaced by uri before /run
    # Use wait=False to get just the task info back
    task = client.run({
        "app": "some/app",
        "input": {"image": str(file_path)},
        "worker_selection_mode": "private",
    }, wait=False)

    # The mocked /run echoes input; ensure it is not the raw path anymore (upload replaced it)
    assert task["input"]["image"] != str(file_path)
    assert task["input"]["image"].startswith("https://cloud.inference.sh/")


def test_upload_file_from_bytes(tmp_path, patch_requests):
    """Test upload_file() with bytes data."""
    client = Inference(api_key="test")

    file_obj = client.upload_file(b"PNGDATA")
    
    assert file_obj["id"] == "file_1"
    assert file_obj["uri"] == "https://cloud.inference.sh/u/user/file_1.png"
    assert len(patch_requests.put_calls) == 1
    assert patch_requests.put_calls[0]["size"] == 7  # len(b"PNGDATA")


def test_upload_file_from_path(tmp_path, patch_requests):
    """Test upload_file() with file path."""
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(b"hello world")

    client = Inference(api_key="test")

    file_obj = client.upload_file(str(file_path))
    
    assert file_obj["id"] == "file_1"
    assert file_obj["uri"] == "https://cloud.inference.sh/u/user/file_1.png"
    assert len(patch_requests.put_calls) == 1
    assert patch_requests.put_calls[0]["size"] == 11  # len(b"hello world")


def test_task_status_enum():
    """Test TaskStatus enum values."""
    assert TaskStatus.RECEIVED == 1
    assert TaskStatus.QUEUED == 2
    assert TaskStatus.SCHEDULED == 3
    assert TaskStatus.PREPARING == 4
    assert TaskStatus.SERVING == 5
    assert TaskStatus.SETTING_UP == 6
    assert TaskStatus.RUNNING == 7
    assert TaskStatus.UPLOADING == 8
    assert TaskStatus.COMPLETED == 9
    assert TaskStatus.FAILED == 10
    assert TaskStatus.CANCELLED == 11


# ==================== Async Tests ====================

class MockAsyncResponse:
    """Mock aiohttp response for async tests."""
    def __init__(self, json_data=None, status=200, lines=None):
        self._json_data = json_data or {"success": True, "data": {}}
        self.status = status
        self._lines = lines or []
    
    async def json(self):
        return self._json_data
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    @property
    def content(self):
        """Async iterator for SSE lines."""
        return MockAsyncIterator(self._lines)


class MockAsyncIterator:
    """Mock async iterator for SSE content."""
    def __init__(self, lines):
        self._lines = iter(lines)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


class MockClientSession:
    """Mock aiohttp.ClientSession."""
    def __init__(self, responses):
        self._responses = responses
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    def request(self, method, url, **kwargs):
        return self._get_response(method, url, kwargs)
    
    def get(self, url, **kwargs):
        return self._get_response("GET", url, kwargs)
    
    def post(self, url, **kwargs):
        return self._get_response("POST", url, kwargs)
    
    def put(self, url, **kwargs):
        return self._get_response("PUT", url, kwargs)
    
    def _get_response(self, method, url, kwargs):
        # Create task
        if url.endswith("/run") and method.upper() == "POST":
            data = kwargs.get("json", {})
            return MockAsyncResponse(json_data={
                "success": True,
                "data": {
                    "id": "task_async_123",
                    "status": 1,
                    "input": data.get("input"),
                },
            })
        
        # Get task
        if "/tasks/task_async_123" in url and method.upper() == "GET" and not url.endswith("/stream"):
            return MockAsyncResponse(json_data={
                "success": True,
                "data": {
                    "id": "task_async_123",
                    "status": 7,  # RUNNING
                    "input": {"text": "hello"},
                },
            })
        
        # SSE stream
        if url.endswith("/tasks/task_async_123/stream") and method.upper() == "GET":
            event_payload = json.dumps({
                "id": "task_async_123",
                "status": 9,  # COMPLETED
                "output": {"async_ok": True},
                "logs": ["async_done"],
            })
            lines = [
                f"data: {event_payload}\n".encode(),
            ]
            return MockAsyncResponse(status=200, lines=lines)
        
        # Cancel
        if url.endswith("/tasks/task_async_123/cancel") and method.upper() == "POST":
            return MockAsyncResponse(json_data={"success": True, "data": None})
        
        # Files create
        if url.endswith("/files") and method.upper() == "POST":
            return MockAsyncResponse(json_data={
                "success": True,
                "data": [
                    {
                        "id": "file_async_1",
                        "uri": "https://cloud.inference.sh/u/user/file_async_1.png",
                        "upload_url": "https://upload.example.com/async_file",
                    }
                ],
            })
        
        # File upload PUT
        if method.upper() == "PUT":
            return MockAsyncResponse(status=200)
        
        return MockAsyncResponse()


@pytest.fixture
def patch_aiohttp(monkeypatch):
    """Patch aiohttp for async tests."""
    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
    mock_aiohttp.ClientSession = lambda **kwargs: MockClientSession({})
    
    async def require_aiohttp():
        return mock_aiohttp
    
    from inferencesh import client as client_mod
    monkeypatch.setattr(client_mod, "_require_aiohttp", require_aiohttp)
    
    return mock_aiohttp


@pytest.mark.asyncio
async def test_async_run_wait_false(patch_aiohttp):
    """Test async run() with wait=False returns task info immediately."""
    client = AsyncInference(api_key="test")
    
    task = await client.run({
        "app": "some/app",
        "input": {"text": "hello"},
    }, wait=False)
    
    assert task["id"] == "task_async_123"
    assert task["status"] == 1  # RECEIVED


@pytest.mark.asyncio
async def test_async_run_wait_true(patch_aiohttp):
    """Test async run() with wait=True (default) waits for completion."""
    client = AsyncInference(api_key="test")
    
    result = await client.run({
        "app": "some/app",
        "input": {"text": "hello"},
    })
    
    assert result["id"] == "task_async_123"
    assert result["output"] == {"async_ok": True}
    assert result["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_async_run_stream(patch_aiohttp):
    """Test async run() with stream=True returns async iterator."""
    client = AsyncInference(api_key="test")
    
    updates = []
    async for update in await client.run({
        "app": "some/app",
        "input": {"text": "hello"},
    }, stream=True):
        updates.append(update)
        if update.get("status") == TaskStatus.COMPLETED:
            break
    
    assert len(updates) >= 1
    final = updates[-1]
    assert final["id"] == "task_async_123"
    assert final["output"] == {"async_ok": True}
    assert final["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_async_get_task(patch_aiohttp):
    """Test async get_task() returns current task state."""
    client = AsyncInference(api_key="test")
    
    task = await client.get_task("task_async_123")
    
    assert task["id"] == "task_async_123"
    assert task["status"] == 7  # RUNNING


@pytest.mark.asyncio
async def test_async_cancel(patch_aiohttp):
    """Test async cancel() cancels a task."""
    client = AsyncInference(api_key="test")
    
    # Should not raise
    await client.cancel("task_async_123")


@pytest.mark.asyncio
async def test_async_stream_task(patch_aiohttp):
    """Test async stream_task() returns AsyncTaskStream."""
    client = AsyncInference(api_key="test")
    
    async with client.stream_task("task_async_123") as stream:
        updates = []
        async for update in stream:
            updates.append(update)
            if update.get("status") == TaskStatus.COMPLETED:
                break
    
    assert len(updates) >= 1
    assert stream.result is not None
    assert stream.result["output"] == {"async_ok": True}
