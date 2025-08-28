from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Generator, Union, Iterator
from dataclasses import dataclass
from enum import IntEnum
import json
import re
import time
import mimetypes
import os
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable


class TaskStream(AbstractContextManager['TaskStream']):
    """A context manager for streaming task updates.
    
    This class provides a Pythonic interface for handling streaming updates from a task.
    It can be used either as a context manager or as an iterator.
    
    Example:
        ```python
        # As a context manager
        with client.stream_task(task_id) as stream:
            for update in stream:
                print(f"Update: {update}")
                
        # As an iterator
        for update in client.stream_task(task_id):
            print(f"Update: {update}")
        ```
    """
    def __init__(
        self,
        task: Dict[str, Any],
        client: Any,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ):
        self.task = task
        self.client = client
        self.task_id = task["id"]
        self.auto_reconnect = auto_reconnect
        self.max_reconnects = max_reconnects
        self.reconnect_delay_ms = reconnect_delay_ms
        self._final_task: Optional[Dict[str, Any]] = None
        self._error: Optional[Exception] = None
        
    def __enter__(self) -> 'TaskStream':
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.stream()
        
    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """The final task result if completed, None otherwise."""
        return self._final_task
        
    @property
    def error(self) -> Optional[Exception]:
        """The error that occurred during streaming, if any."""
        return self._error
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream updates for this task.
        
        Yields:
            Dict[str, Any]: Task update events
            
        Raises:
            RuntimeError: If the task fails or is cancelled
        """
        try:
            for update in self.client._stream_updates(
                self.task_id,
                self.task,
            ):
                if isinstance(update, Exception):
                    self._error = update
                    raise update
                if update.get("status") == TaskStatus.COMPLETED:
                    self._final_task = update
                yield update
        except Exception as exc:
            self._error = exc
            raise


@runtime_checkable
class TaskCallback(Protocol):
    """Protocol for task streaming callbacks."""
    def on_update(self, data: Dict[str, Any]) -> None:
        """Called when a task update is received."""
        ...

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs during task execution."""
        ...

    def on_complete(self, task: Dict[str, Any]) -> None:
        """Called when a task completes successfully."""
        ...


# Deliberately do lazy imports for requests/aiohttp to avoid hard dependency at import time
def _require_requests():
    try:
        import requests  # type: ignore
        return requests
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "The 'requests' package is required for synchronous HTTP calls. Install with: pip install requests"
        ) from exc


async def _require_aiohttp():
    try:
        import aiohttp  # type: ignore
        return aiohttp
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "The 'aiohttp' package is required for async HTTP calls. Install with: pip install aiohttp"
        ) from exc


class TaskStatus(IntEnum):
    RECEIVED = 1
    QUEUED = 2
    SCHEDULED = 3
    PREPARING = 4
    SERVING = 5
    SETTING_UP = 6
    RUNNING = 7
    UPLOADING = 8
    COMPLETED = 9
    FAILED = 10
    CANCELLED = 11


Base64_RE = re.compile(r"^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$")


@dataclass
class UploadFileOptions:
    filename: Optional[str] = None
    content_type: Optional[str] = None
    path: Optional[str] = None
    public: Optional[bool] = None


class StreamManager:
    """Simple SSE stream manager with optional auto-reconnect."""

    def __init__(
        self,
        *,
        create_event_source: Callable[[], Any],
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_data: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._create_event_source = create_event_source
        self._auto_reconnect = auto_reconnect
        self._max_reconnects = max_reconnects
        self._reconnect_delay_ms = reconnect_delay_ms
        self._on_error = on_error
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_data = on_data

        self._stopped = False
        self._reconnect_attempts = 0
        self._had_successful_connection = False

    def stop(self) -> None:
        self._stopped = True
        if self._on_stop:
            self._on_stop()

    def connect(self) -> None:
        self._stopped = False
        self._reconnect_attempts = 0
        while not self._stopped:
            try:
                if self._on_start:
                    self._on_start()
                event_source = self._create_event_source()
                try:
                    for data in event_source:
                        if self._stopped:
                            break
                        self._had_successful_connection = True
                        if self._on_data:
                            self._on_data(data)
                            # Check again after processing in case on_data stopped us
                            if self._stopped:
                                break
                finally:
                    # Clean up the event source if it has a close method
                    try:
                        if hasattr(event_source, 'close'):
                            event_source.close()
                    except Exception:
                        raise

                # If we're stopped or don't want to auto-reconnect, break immediately
                if self._stopped or not self._auto_reconnect:
                    break
            except Exception as exc:  # noqa: BLE001
                if self._on_error:
                    self._on_error(exc)
                if self._stopped:
                    break
                # If never connected and exceeded attempts, stop
                if not self._had_successful_connection:
                    self._reconnect_attempts += 1
                    if self._reconnect_attempts > self._max_reconnects:
                        break
                time.sleep(self._reconnect_delay_ms / 1000.0)
            else:
                # Completed without exception - if we want to auto-reconnect only after success
                if not self._auto_reconnect:
                    break
                time.sleep(self._reconnect_delay_ms / 1000.0)


class Inference:
    """Synchronous client for inference.sh API, mirroring the JS SDK behavior.

    Args:
        api_key (str): The API key for authentication
        base_url (Optional[str]): Override the default API base URL
        sse_chunk_size (Optional[int]): Chunk size for SSE reading (default: 8192 bytes)
        sse_mode (Optional[str]): SSE reading mode ('iter_lines' or 'raw', default: 'iter_lines')

    The client supports performance tuning for SSE (Server-Sent Events) through:
    1. sse_chunk_size: Controls the buffer size for reading SSE data (default: 8KB)
       - Larger values may improve performance but use more memory
       - Can also be set via INFERENCE_SSE_READ_BYTES environment variable
    2. sse_mode: Controls how SSE data is read ('iter_lines' or 'raw')
       - 'iter_lines': Uses requests' built-in line iteration (default)
       - 'raw': Uses lower-level socket reading
       - Can also be set via INFERENCE_SSE_MODE environment variable
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        sse_chunk_size: Optional[int] = None,
        sse_mode: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url or "https://api.inference.sh"

        # SSE configuration with environment variable fallbacks
        self._sse_mode = sse_mode or os.getenv("INFERENCE_SSE_MODE") or "iter_lines"
        self._sse_mode = self._sse_mode.lower()

        # Default to 8KB chunks, can be overridden by parameter or env var
        try:
            env_chunk_size = os.getenv("INFERENCE_SSE_READ_BYTES")
            if sse_chunk_size is not None:
                self._sse_read_bytes = sse_chunk_size
            elif env_chunk_size is not None:
                self._sse_read_bytes = int(env_chunk_size)
            else:
                self._sse_read_bytes = 8192  # 8KB default
        except Exception:
            self._sse_read_bytes = 8192  # Default to 8KB chunks on error

    # --------------- HTTP helpers ---------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        requests = _require_requests()
        url = f"{self._base_url}{endpoint}"
        merged_headers = {**self._headers(), **(headers or {})}
        resp = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            data=json.dumps(data) if data is not None else None,
            headers=merged_headers,
            stream=stream,
            timeout=timeout or 30,
        )
        if stream:
            return resp
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or not payload.get("success", False):
            message = None
            if isinstance(payload, dict) and payload.get("error"):
                err = payload["error"]
                if isinstance(err, dict):
                    message = err.get("message")
                else:
                    message = str(err)
            raise RuntimeError(message or "Request failed")
        return payload.get("data")

    # --------------- Public API ---------------
    def run(
        self,
        params: Dict[str, Any],
        *,
        wait: bool = True,
        stream: bool = False,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> Union[Dict[str, Any], TaskStream, Iterator[Dict[str, Any]]]:
        """Run a task with optional streaming updates.
        
        By default, this method waits for the task to complete and returns the final result.
        You can set wait=False to get just the task info, or stream=True to get an iterator
        of status updates.
        
        Args:
            params: Task parameters to pass to the API
            wait: Whether to wait for task completion (default: True)
            stream: Whether to return an iterator of updates (default: False)
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnects: Maximum number of reconnection attempts
            reconnect_delay_ms: Delay between reconnection attempts in milliseconds
            
        Returns:
            Union[Dict[str, Any], TaskStream, Iterator[Dict[str, Any]]]:
                - If wait=True and stream=False: The completed task data
                - If wait=False: The created task info
                - If stream=True: An iterator of task updates
            
        Example:
            ```python
            # Simple usage - wait for result (default)
            result = client.run(params)
            print(f"Output: {result['output']}")
            
            # Get task info without waiting
            task = client.run(params, wait=False)
            task_id = task["id"]
            
            # Stream updates
            for update in client.run(params, stream=True):
                print(f"Status: {update.get('status')}")
                if update.get('status') == TaskStatus.COMPLETED:
                    print(f"Result: {update.get('output')}")
            ```
        """
        # Create the task
        processed_input = self._process_input_data(params.get("input"))
        task = self._request("post", "/run", data={**params, "input": processed_input})
        
        # Return immediately if not waiting
        if not wait and not stream:
            return _strip_task(task)
            
        # Return stream if requested
        if stream:
            task_stream = TaskStream(
                task=task,
                client=self,
                auto_reconnect=auto_reconnect,
                max_reconnects=max_reconnects,
                reconnect_delay_ms=reconnect_delay_ms,
            )
            return task_stream
            
        # Otherwise wait for completion
        return self.wait_for_completion(task["id"])



    def cancel(self, task_id: str) -> None:
        self._request("post", f"/tasks/{task_id}/cancel")

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get the current state of a task.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            Dict[str, Any]: The current task state
        """
        return self._request("get", f"/tasks/{task_id}")

    def wait_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for a task to complete and return its final state.
        
        This method polls the task status until it reaches a terminal state
        (completed, failed, or cancelled).
        
        Args:
            task_id: The ID of the task to wait for
            
        Returns:
            Dict[str, Any]: The final task state
            
        Raises:
            RuntimeError: If the task fails or is cancelled
        """
        with self.stream_task(task_id) as stream:
            for update in stream:
                if update.get("status") == TaskStatus.COMPLETED:
                    return update
                elif update.get("status") == TaskStatus.FAILED:
                    raise RuntimeError(update.get("error") or "Task failed")
                elif update.get("status") == TaskStatus.CANCELLED:
                    raise RuntimeError("Task cancelled")
        raise RuntimeError("Stream ended without completion")

    # --------------- File upload ---------------
    def upload_file(self, data: Union[str, bytes], options: Optional[UploadFileOptions] = None) -> Dict[str, Any]:
        options = options or UploadFileOptions()
        content_type = options.content_type
        raw_bytes: bytes
        if isinstance(data, bytes):
            raw_bytes = data
            if not content_type:
                content_type = "application/octet-stream"
        else:
            # Prefer local filesystem path if it exists
            if os.path.exists(data):
                path = data
                guessed = mimetypes.guess_type(path)[0]
                content_type = content_type or guessed or "application/octet-stream"
                with open(path, "rb") as f:
                    raw_bytes = f.read()
                if not options.filename:
                    options.filename = os.path.basename(path)
            elif data.startswith("data:"):
                # data URI
                match = re.match(r"^data:([^;]+);base64,(.+)$", data)
                if not match:
                    raise ValueError("Invalid base64 data URI format")
                content_type = content_type or match.group(1)
                raw_bytes = _b64_to_bytes(match.group(2))
            elif _looks_like_base64(data):
                raw_bytes = _b64_to_bytes(data)
                content_type = content_type or "application/octet-stream"
            else:
                raise ValueError("upload_file expected bytes, data URI, base64 string, or existing file path")

        file_req = {
            "files": [
                {
                    "uri": "",
                    "filename": options.filename,
                    "content_type": content_type,
                    "path": options.path,
                    "size": len(raw_bytes),
                    "public": options.public,
                }
            ]
        }

        created = self._request("post", "/files", data=file_req)
        file_obj = created[0]
        upload_url = file_obj.get("upload_url")
        if not upload_url:
            raise RuntimeError("No upload URL provided by the server")

        # Upload to S3 (or compatible) signed URL
        requests = _require_requests()
        put_resp = requests.put(upload_url, data=raw_bytes, headers={"Content-Type": content_type})
        if not (200 <= put_resp.status_code < 300):
            raise RuntimeError(f"Failed to upload file content: {put_resp.reason}")
        return file_obj

    # --------------- Helpers ---------------
    def stream_task(
        self,
        task_id: str,
        *,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> TaskStream:
        """Create a TaskStream for getting streaming updates from a task.
        
        This provides a more Pythonic interface for handling task updates compared to callbacks.
        The returned TaskStream can be used either as a context manager or as an iterator.
        
        Args:
            task_id: The ID of the task to stream
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnects: Maximum number of reconnection attempts
            reconnect_delay_ms: Delay between reconnection attempts in milliseconds
            
        Returns:
            TaskStream: A stream interface for the task
            
        Example:
            ```python
            # Run a task
            task = client.run(params)
            
            # Stream updates using context manager
            with client.stream_task(task["id"]) as stream:
                for update in stream:
                    print(f"Status: {update.get('status')}")
                    if update.get("status") == TaskStatus.COMPLETED:
                        print(f"Result: {update.get('output')}")
                        
            # Or use as a simple iterator
            for update in client.stream_task(task["id"]):
                print(f"Update: {update}")
            ```
        """
        task = self.get_task(task_id)
        return TaskStream(
            task=task,
            client=self,
            auto_reconnect=auto_reconnect,
            max_reconnects=max_reconnects,
            reconnect_delay_ms=reconnect_delay_ms,
        )

    def _stream_updates(
        self,
        task_id: str,
        task: Dict[str, Any],
    ) -> Generator[Union[Dict[str, Any], Exception], None, None]:
        """Internal method to stream task updates."""
        url = f"/tasks/{task_id}/stream"
        resp = self._request(
            "get",
            url,
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Accept-Encoding": "identity",
                "Connection": "keep-alive",
            },
            stream=True,
            timeout=60,
        )
        try:
            for evt in self._iter_sse(resp):
                try:
                    # Process the event to check for completion/errors
                    result = _process_stream_event(
                        evt,
                        task=task,
                        stopper=None,  # We'll handle stopping via the iterator
                    )
                    if result is not None:
                        yield result
                        break
                    yield _strip_task(evt)
                except Exception as exc:
                    yield exc
                    raise
        finally:
            try:
                # Force close the underlying socket if possible
                try:
                    raw = getattr(resp, 'raw', None)
                    if raw is not None:
                        raw.close()
                except Exception:
                    raise
                # Close the response
                resp.close()
            except Exception:
                raise

    def _iter_sse(self, resp: Any, stream_manager: Optional[Any] = None) -> Generator[Dict[str, Any], None, None]:
        """Iterate JSON events from an SSE response."""
        # Mode 1: raw socket readline (can reduce buffering in some environments)
        if self._sse_mode == "raw":
            raw = getattr(resp, "raw", None)
            if raw is not None:
                try:
                    # Avoid urllib3 decompression buffering
                    raw.decode_content = False  # type: ignore[attr-defined]
                except Exception:
                    raise
                buf = bytearray()
                read_size = max(1, int(self._sse_read_bytes))
                while True:
                    # Check if we've been asked to stop before reading more data
                    try:
                        if stream_manager and stream_manager._stopped:  # type: ignore[attr-defined]
                            break
                    except Exception:
                        raise

                    chunk = raw.read(read_size)
                    if not chunk:
                        break
                    for b in chunk:
                        if b == 10:  # '\n'
                            try:
                                line = buf.decode(errors="ignore").rstrip("\r")
                            except Exception:
                                line = ""
                            buf.clear()
                            if not line:
                                continue
                            if line.startswith(":"):
                                continue
                            if line.startswith("data:"):
                                data_str = line[5:].lstrip()
                                if not data_str:
                                    continue
                                try:
                                    yield json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue
                        else:
                            buf.append(b)
                return
        # Mode 2: default iter_lines with reasonable chunk size (8KB)
        for line in resp.iter_lines(decode_unicode=True, chunk_size=8192):
            # Check if we've been asked to stop before processing any more lines
            try:
                if stream_manager and stream_manager._stopped:  # type: ignore[attr-defined]
                    break
            except Exception:
                raise

            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_str = line[5:].lstrip()
                if not data_str:
                    continue
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue

    def _process_input_data(self, input_value: Any, path: str = "root") -> Any:
        if input_value is None:
            return input_value

        # Handle lists
        if isinstance(input_value, list):
            return [self._process_input_data(item, f"{path}[{idx}]") for idx, item in enumerate(input_value)]

        # Handle dicts
        if isinstance(input_value, dict):
            processed: Dict[str, Any] = {}
            for key, value in input_value.items():
                processed[key] = self._process_input_data(value, f"{path}.{key}")
            return processed

        # Handle strings that are filesystem paths, data URIs, or base64
        if isinstance(input_value, str):
            # Prefer existing local file paths first to avoid misclassifying plain strings
            if os.path.exists(input_value):
                file_obj = self.upload_file(input_value)
                return file_obj.get("uri")
            if input_value.startswith("data:") or _looks_like_base64(input_value):
                file_obj = self.upload_file(input_value)
                return file_obj.get("uri")
            return input_value

        # Handle File-like objects from our models
        try:
            from .models.file import File as SDKFile  # local import to avoid cycle
            if isinstance(input_value, SDKFile):
                # Prefer local path if present, else uri
                src = input_value.path or input_value.uri
                if not src:
                    return input_value
                file_obj = self.upload_file(src, UploadFileOptions(filename=input_value.filename, content_type=input_value.content_type))
                return file_obj.get("uri")
        except Exception:
            raise

        return input_value


class AsyncInference:
    """Async client for inference.sh API, mirroring the JS SDK behavior."""

    def __init__(self, *, api_key: str, base_url: Optional[str] = None) -> None:
        self._api_key = api_key
        self._base_url = base_url or "https://api.inference.sh"

    # --------------- HTTP helpers ---------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        expect_stream: bool = False,
    ) -> Any:
        aiohttp = await _require_aiohttp()
        url = f"{self._base_url}{endpoint}"
        merged_headers = {**self._headers(), **(headers or {})}
        timeout_cfg = aiohttp.ClientTimeout(total=timeout or 30)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data,
                headers=merged_headers,
            ) as resp:
                if expect_stream:
                    return resp
                payload = await resp.json()
                if not isinstance(payload, dict) or not payload.get("success", False):
                    message = None
                    if isinstance(payload, dict) and payload.get("error"):
                        err = payload["error"]
                        if isinstance(err, dict):
                            message = err.get("message")
                        else:
                            message = str(err)
                    raise RuntimeError(message or "Request failed")
                return payload.get("data")

    # --------------- Public API ---------------
    async def run(
        self,
        params: Dict[str, Any],
        *,
        wait: bool = True,
        stream: bool = False,
        auto_reconnect: bool = True,
        max_reconnects: int = 5,
        reconnect_delay_ms: int = 1000,
    ) -> Union[Dict[str, Any], TaskStream, Iterator[Dict[str, Any]]]:
        """Run a task with optional streaming updates.
        
        By default, this method waits for the task to complete and returns the final result.
        You can set wait=False to get just the task info, or stream=True to get an iterator
        of status updates.
        
        Args:
            params: Task parameters to pass to the API
            wait: Whether to wait for task completion (default: True)
            stream: Whether to return an iterator of updates (default: False)
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnects: Maximum number of reconnection attempts
            reconnect_delay_ms: Delay between reconnection attempts in milliseconds
            
        Returns:
            Union[Dict[str, Any], TaskStream, Iterator[Dict[str, Any]]]:
                - If wait=True and stream=False: The completed task data
                - If wait=False: The created task info
                - If stream=True: An iterator of task updates
            
        Example:
            ```python
            # Simple usage - wait for result (default)
            result = await client.run(params)
            print(f"Output: {result['output']}")
            
            # Get task info without waiting
            task = await client.run(params, wait=False)
            task_id = task["id"]
            
            # Stream updates
            async for update in await client.run(params, stream=True):
                print(f"Status: {update.get('status')}")
                if update.get('status') == TaskStatus.COMPLETED:
                    print(f"Result: {update.get('output')}")
            ```
        """
        # Create the task
        processed_input = await self._process_input_data(params.get("input"))
        task = await self._request("post", "/run", data={**params, "input": processed_input})
        
        # Return immediately if not waiting
        if not wait and not stream:
            return task
            
        # Return stream if requested
        if stream:
            task_stream = TaskStream(
                task=task,
                client=self,
                auto_reconnect=auto_reconnect,
                max_reconnects=max_reconnects,
                reconnect_delay_ms=reconnect_delay_ms,
            )
            return task_stream
            
        # Otherwise wait for completion
        return await self.wait_for_completion(task["id"])

    async def cancel(self, task_id: str) -> None:
        await self._request("post", f"/tasks/{task_id}/cancel")

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get the current state of a task.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            Dict[str, Any]: The current task state
        """
        return await self._request("get", f"/tasks/{task_id}")

    async def wait_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for a task to complete and return its final state.
        
        This method polls the task status until it reaches a terminal state
        (completed, failed, or cancelled).
        
        Args:
            task_id: The ID of the task to wait for
            
        Returns:
            Dict[str, Any]: The final task state
            
        Raises:
            RuntimeError: If the task fails or is cancelled
        """
        with self.stream_task(task_id) as stream:
            async for update in stream:
                if update.get("status") == TaskStatus.COMPLETED:
                    return update
                elif update.get("status") == TaskStatus.FAILED:
                    raise RuntimeError(update.get("error") or "Task failed")
                elif update.get("status") == TaskStatus.CANCELLED:
                    raise RuntimeError("Task cancelled")
        raise RuntimeError("Stream ended without completion")

    # --------------- File upload ---------------
    async def upload_file(self, data: Union[str, bytes], options: Optional[UploadFileOptions] = None) -> Dict[str, Any]:
        options = options or UploadFileOptions()
        content_type = options.content_type
        raw_bytes: bytes
        if isinstance(data, bytes):
            raw_bytes = data
            if not content_type:
                content_type = "application/octet-stream"
        else:
            if os.path.exists(data):
                path = data
                guessed = mimetypes.guess_type(path)[0]
                content_type = content_type or guessed or "application/octet-stream"
                async with await _aio_open_file(path, "rb") as f:
                    raw_bytes = await f.read()  # type: ignore[attr-defined]
                if not options.filename:
                    options.filename = os.path.basename(path)
            elif data.startswith("data:"):
                match = re.match(r"^data:([^;]+);base64,(.+)$", data)
                if not match:
                    raise ValueError("Invalid base64 data URI format")
                content_type = content_type or match.group(1)
                raw_bytes = _b64_to_bytes(match.group(2))
            elif _looks_like_base64(data):
                raw_bytes = _b64_to_bytes(data)
                content_type = content_type or "application/octet-stream"
            else:
                raise ValueError("upload_file expected bytes, data URI, base64 string, or existing file path")

        file_req = {
            "files": [
                {
                    "uri": "",
                    "filename": options.filename,
                    "content_type": content_type,
                    "path": options.path,
                    "size": len(raw_bytes),
                    "public": options.public,
                }
            ]
        }

        created = await self._request("post", "/files", data=file_req)
        file_obj = created[0]
        upload_url = file_obj.get("upload_url")
        if not upload_url:
            raise RuntimeError("No upload URL provided by the server")

        aiohttp = await _require_aiohttp()
        timeout_cfg = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.put(upload_url, data=raw_bytes, headers={"Content-Type": content_type}) as resp:
                if resp.status // 100 != 2:
                    raise RuntimeError(f"Failed to upload file content: {resp.reason}")
        return file_obj

    # --------------- Helpers ---------------
    async def _process_input_data(self, input_value: Any, path: str = "root") -> Any:
        if input_value is None:
            return input_value

        if isinstance(input_value, list):
            return [await self._process_input_data(item, f"{path}[{idx}]") for idx, item in enumerate(input_value)]

        if isinstance(input_value, dict):
            processed: Dict[str, Any] = {}
            for key, value in input_value.items():
                processed[key] = await self._process_input_data(value, f"{path}.{key}")
            return processed

        if isinstance(input_value, str):
            if os.path.exists(input_value):
                file_obj = await self.upload_file(input_value)
                return file_obj.get("uri")
            if input_value.startswith("data:") or _looks_like_base64(input_value):
                file_obj = await self.upload_file(input_value)
                return file_obj.get("uri")
            return input_value

        try:
            from .models.file import File as SDKFile  # local import
            if isinstance(input_value, SDKFile):
                src = input_value.path or input_value.uri
                if not src:
                    return input_value
                file_obj = await self.upload_file(src, UploadFileOptions(filename=input_value.filename, content_type=input_value.content_type))
                return file_obj.get("uri")
        except Exception:
            raise

        return input_value

    async def _aiter_sse(self, resp: Any) -> Generator[Dict[str, Any], None, None]:
        async for raw_line in resp.content:  # type: ignore[attr-defined]
            try:
                line = raw_line.decode().rstrip("\n")
            except Exception:
                continue
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_str = line[5:].lstrip()
                if not data_str:
                    continue
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue


# --------------- small async utilities ---------------
async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _b64_to_bytes(b64: str) -> bytes:
    import base64

    return base64.b64decode(b64)


async def _aio_open_file(path: str, mode: str):
    import aiofiles  # type: ignore

    return await aiofiles.open(path, mode)


def _looks_like_base64(value: str) -> bool:
    # Reject very short strings to avoid matching normal words like "hi"
    if len(value) < 16:
        return False
    # Quick charset check
    if not Base64_RE.match(value):
        return False
    # Must be divisible by 4
    if len(value) % 4 != 0:
        return False
    # Try decode to be sure
    try:
        _ = _b64_to_bytes(value)
        return True
    except Exception:
        return False


def _strip_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Strip task to essential fields."""
    return {
        "id": task.get("id"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "input": task.get("input"),
        "output": task.get("output"),
        "logs": task.get("logs"),
        "status": task.get("status"),
    }

def _process_stream_event(
    data: Dict[str, Any], *, task: Dict[str, Any], stopper: Optional[Callable[[], None]] = None
) -> Optional[Dict[str, Any]]:
    """Shared handler for SSE task events. Returns final task dict when completed, else None.
    If stopper is provided, it will be called on terminal events to end streaming.
    """
    status = data.get("status")

    if status == TaskStatus.COMPLETED:
        result = _strip_task(data)
        if stopper:
            stopper()
        return result
    if status == TaskStatus.FAILED:
        if stopper:
            stopper()
        raise RuntimeError(data.get("error") or "task failed")
    if status == TaskStatus.CANCELLED:
        if stopper:
            stopper()
        raise RuntimeError("task cancelled")
    return None


